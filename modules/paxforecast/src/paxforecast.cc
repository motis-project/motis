#include "motis/paxforecast/paxforecast.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>

#include "fmt/format.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"
#include "motis/pair.h"
#include "motis/vector.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/station_access.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/message.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/debug.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/paxmon_data.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/error.h"
#include "motis/paxforecast/load_forecast.h"
#include "motis/paxforecast/measures/affected_groups.h"
#include "motis/paxforecast/measures/measures.h"
#include "motis/paxforecast/measures/storage.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/statistics.h"

#include "motis/paxforecast/behavior/default_behavior.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;
using namespace motis::paxmon;

namespace motis::paxforecast {

paxforecast::paxforecast()
    : module("Passenger Forecast", "paxforecast"),
      measures_storage_(std::make_unique<measures::storage>()) {
  param(forecast_filename_, "forecast_results",
        "output file for forecast messages");
  param(behavior_stats_filename_, "behavior_stats",
        "output file for behavior statistics");
  param(routing_cache_filename_, "routing_cache",
        "optional cache file for routing queries");
  param(calc_load_forecast_, "calc_load_forecast",
        "calculate load forecast (required for output/publish)");
  param(publish_load_forecast_, "publish_load_forecast",
        "publish load forecast");
  param(stats_file_, "stats", "statistics file");
  param(deterministic_mode_, "deterministic_mode",
        "all passengers always pick the best alternative");
  param(min_delay_improvement_, "min_delay_improvement",
        "minimum required arrival time improvement for major delay "
        "alternatives (minutes)");
}

paxforecast::~paxforecast() = default;

void paxforecast::init(motis::module::registry& reg) {
  stats_writer_ = std::make_unique<stats_writer>(stats_file_);

  if (!forecast_filename_.empty()) {
    forecast_file_.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    forecast_file_.open(forecast_filename_);
  }

  if (!behavior_stats_filename_.empty()) {
    behavior_stats_file_.exceptions(std::ios_base::failbit |
                                    std::ios_base::badbit);
    behavior_stats_file_.open(behavior_stats_filename_);
    behavior_stats_file_ << "system_time,group_count,cpg_count,"
                         << "found_alt_count_avg,picked_alt_count_avg,"
                         << "best_alt_prob_avg,second_alt_prob_avg\n";
  }

  if (!routing_cache_filename_.empty()) {
    routing_cache_.open(routing_cache_filename_);
  }

  reg.subscribe("/paxmon/monitoring_update",
                [&](msg_ptr const& msg) {
                  on_monitoring_event(msg);
                  return nullptr;
                },
                {});

  reg.subscribe("/paxmon/universe_forked",
                [&](msg_ptr const& msg) {
                  auto const ev = motis_content(PaxMonUniverseForked, msg);
                  LOG(info)
                      << "paxforecast: /paxmon/universe_forked: new="
                      << ev->new_universe() << ", base=" << ev->base_universe();
                  measures_storage_->universe_created(ev->new_universe());
                  return nullptr;
                },
                {});

  reg.subscribe("/paxmon/universe_destroyed",
                [&](msg_ptr const& msg) {
                  auto const ev = motis_content(PaxMonUniverseDestroyed, msg);
                  LOG(info) << "paxforecast: /paxmon/universe_destroyed: "
                            << ev->universe();
                  measures_storage_->universe_destroyed(ev->universe());
                  return nullptr;
                },
                {});

  reg.register_op(
      "/paxforecast/apply_measures",
      [&](msg_ptr const& msg) -> msg_ptr { return apply_measures(msg); }, {});
}

auto const constexpr REMOVE_GROUPS_BATCH_SIZE = 10'000;
auto const constexpr ADD_GROUPS_BATCH_SIZE = 10'000;

void send_remove_groups(universe const& uv,
                        std::vector<std::uint64_t>& groups_to_remove,
                        tick_statistics& tick_stats) {
  if (groups_to_remove.empty()) {
    return;
  }
  tick_stats.removed_groups_ += groups_to_remove.size();
  message_creator remove_groups_mc;
  remove_groups_mc.create_and_finish(
      MsgContent_PaxMonRemoveGroupsRequest,
      CreatePaxMonRemoveGroupsRequest(
          remove_groups_mc, uv.id_,
          remove_groups_mc.CreateVector(groups_to_remove))
          .Union(),
      "/paxmon/remove_groups");
  auto const remove_msg = make_msg(remove_groups_mc);
  motis_call(remove_msg)->val();
  groups_to_remove.clear();
}

void update_tracked_groups(
    schedule const& sched, universe const& uv,
    simulation_result const& sim_result,
    std::map<passenger_group const*, monitoring_event_type> const&
        pg_event_types,
    tick_statistics& tick_stats) {
  using namespace flatbuffers;

  message_creator add_groups_mc;
  auto groups_to_remove = std::vector<std::uint64_t>{};
  auto groups_to_add = std::vector<Offset<PaxMonGroup>>{};
  auto remove_group_count = 0ULL;
  auto add_group_count = 0ULL;

  auto const system_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);

  auto const send_add_groups = [&]() {
    if (groups_to_add.empty()) {
      return;
    }
    add_groups_mc.create_and_finish(
        MsgContent_PaxMonAddGroupsRequest,
        CreatePaxMonAddGroupsRequest(add_groups_mc, uv.id_,
                                     add_groups_mc.CreateVector(groups_to_add))
            .Union(),
        "/paxmon/add_groups");
    auto const add_msg = make_msg(add_groups_mc);
    motis_call(add_msg)->val();
    groups_to_add.clear();
    add_groups_mc.Clear();
  };

  for (auto const& [pg, result] : sim_result.group_results_) {
    if (result.alternatives_.empty()) {
      // keep existing group (only reachable part)
      continue;
    }

    auto event_type = monitoring_event_type::NO_PROBLEM;
    if (auto const it = pg_event_types.find(pg); it != end(pg_event_types)) {
      event_type = it->second;
    }

    auto const journey_prefix =
        get_prefix(sched, pg->compact_planned_journey_, *result.localization_);

    // major delay groups have already been removed
    if (event_type != monitoring_event_type::MAJOR_DELAY_EXPECTED) {
      // remove existing group
      groups_to_remove.emplace_back(pg->id_);
      ++remove_group_count;
    }

    // add alternatives
    for (auto const& [alt, prob] : result.alternatives_) {
      if (prob == 0.0) {
        continue;
      }

      compact_journey new_journey;
      try {
        new_journey =
            merge_journeys(sched, journey_prefix, alt->compact_journey_);
      } catch (std::runtime_error const& e) {
        std::cout << "\noriginal planned journey:\n";
        for (auto const& leg : pg->compact_planned_journey_.legs_) {
          print_leg(sched, leg);
        }
        std::cout << "\nlocalization: in_trip="
                  << result.localization_->in_trip()
                  << ", first_station=" << result.localization_->first_station_
                  << ", station="
                  << result.localization_->at_station_->name_.str()
                  << ", schedule_arrival_time="
                  << format_time(result.localization_->schedule_arrival_time_)
                  << ", current_arrival_time="
                  << format_time(result.localization_->current_arrival_time_)
                  << "\n";
        if (result.localization_->in_trip()) {
          print_trip(sched, result.localization_->in_trip_);
        }
        throw e;
      }

      groups_to_add.emplace_back(to_fbs(
          sched, add_groups_mc,
          make_passenger_group(std::move(new_journey), pg->source_,
                               pg->passengers_, pg->planned_arrival_time_,
                               pg->source_flags_ | group_source_flags::FORECAST,
                               prob, system_time, pg->id_,
                               pg->generation_ + 1)));
      ++add_group_count;
    }

    if (groups_to_remove.size() >= REMOVE_GROUPS_BATCH_SIZE) {
      send_remove_groups(uv, groups_to_remove, tick_stats);
    }
    if (groups_to_add.size() >= ADD_GROUPS_BATCH_SIZE) {
      send_add_groups();
    }
  }

  LOG(info) << "update_tracked_groups: -" << remove_group_count << " +"
            << add_group_count;
  tick_stats.added_groups_ += add_group_count;

  send_remove_groups(uv, groups_to_remove, tick_stats);
  send_add_groups();
}

bool has_better_alternative(std::vector<alternative> const& alts,
                            time expected_arrival_time,
                            duration min_improvement) {
  auto const latest_accepted_arrival = expected_arrival_time - min_improvement;
  return std::any_of(begin(alts), end(alts),
                     [latest_accepted_arrival](alternative const& alt) {
                       return alt.arrival_time_ <= latest_accepted_arrival;
                     });
}

void paxforecast::on_monitoring_event(msg_ptr const& msg) {
  auto const mon_update = motis_content(PaxMonUpdate, msg);
  MOTIS_START_TIMING(total);
  auto& data =
      *get_shared_data<paxmon_data*>(to_res_id(global_res_id::PAX_DATA));
  auto const uv_access = get_universe_and_schedule(data, mon_update->universe(),
                                                   ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto& caps = data.capacity_maps_;

  tick_statistics tick_stats;
  tick_stats.system_time_ = sched.system_time_;

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME,
              "paxforecast::on_monitoring_event: invalid current system time: "
              "system_time={}, schedule_begin={}",
              sched.system_time_, sched.schedule_begin_);

  std::map<unsigned, std::vector<combined_passenger_group>> combined_groups;
  std::map<passenger_group const*, monitoring_event_type> pg_event_types;
  std::map<passenger_group const*, time> expected_arrival_times;
  auto delayed_groups = 0ULL;

  for (auto const& event : *mon_update->events()) {
    if (event->type() == PaxMonEventType_NO_PROBLEM) {
      continue;
    }

    auto const pg = uv.get_passenger_group(event->group()->id());
    utl::verify(pg != nullptr, "monitored passenger group already removed");

    auto const major_delay =
        event->type() == PaxMonEventType_MAJOR_DELAY_EXPECTED;

    if (major_delay) {
      ++delayed_groups;
      expected_arrival_times.insert(
          {pg, unix_to_motistime(sched.schedule_begin_,
                                 event->expected_arrival_time())});
    }

    auto const inserted = pg_event_types.insert(
        {pg, static_cast<monitoring_event_type>(event->type())});
    utl::verify(inserted.second,
                "multiple monitoring updates for passenger group");
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id =
        pg->compact_planned_journey_.destination_station_id();

    auto& destination_groups = combined_groups[destination_station_id];
    auto cpg = std::find_if(
        begin(destination_groups), end(destination_groups),
        [&](auto const& g) { return g.localization_ == localization; });
    if (cpg == end(destination_groups)) {
      destination_groups.emplace_back(
          combined_passenger_group{destination_station_id,
                                   pg->passengers_,
                                   major_delay,
                                   localization,
                                   {pg},
                                   {}});
    } else {
      cpg->passengers_ += pg->passengers_;
      cpg->groups_.push_back(pg);
      if (major_delay) {
        cpg->has_major_delay_groups_ = true;
      }
    }
  }

  if (combined_groups.empty()) {
    return;
  }

  LOG(info) << mon_update->events()->size() << " monitoring updates, "
            << pg_event_types.size() << " groups, " << combined_groups.size()
            << " combined groups";

  tick_stats.monitoring_events_ = mon_update->events()->size();
  tick_stats.groups_ = pg_event_types.size();
  tick_stats.combined_groups_ = combined_groups.size();
  tick_stats.major_delay_groups_ = delayed_groups;

  auto routing_requests = 0ULL;
  auto alternatives_found = 0ULL;

  {
    MOTIS_START_TIMING(find_alternatives);
    scoped_timer alt_timer{"on_monitoring_event: find alternatives"};
    std::vector<ctx::future_ptr<ctx_data, void>> futures;
    for (auto& cgs : combined_groups) {
      auto const destination_station_id = cgs.first;
      for (auto& cpg : cgs.second) {
        ++routing_requests;
        futures.emplace_back(
            spawn_job_void([this, &uv, &sched, destination_station_id, &cpg] {
              cpg.alternatives_ = find_alternatives(
                  uv, sched, routing_cache_, {}, destination_station_id,
                  cpg.localization_, nullptr, true, 0);
            }));
      }
    }
    LOG(info) << "find alternatives: " << routing_requests
              << " routing requests (using cache=" << routing_cache_.is_open()
              << ")...";
    ctx::await_all(futures);
    routing_cache_.sync();
    MOTIS_STOP_TIMING(find_alternatives);
    tick_stats.t_find_alternatives_ = MOTIS_TIMING_MS(find_alternatives);
  }

  {
    MOTIS_START_TIMING(add_alternatives);
    scoped_timer alt_trips_timer{"add alternatives to graph"};
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        alternatives_found += cpg.alternatives_.size();
        for (auto const& alt : cpg.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, caps, uv, leg.trip_idx_);
          }
        }
      }
    }
    MOTIS_STOP_TIMING(add_alternatives);
    tick_stats.t_add_alternatives_ = MOTIS_TIMING_MS(add_alternatives);
  }

  LOG(info) << "alternatives: " << routing_requests << " routing requests => "
            << alternatives_found << " alternatives";

  tick_stats.routing_requests_ = routing_requests;
  tick_stats.alternatives_found_ = alternatives_found;

  std::vector<std::unique_ptr<passenger_group>> removed_groups;
  auto removed_group_count = 0ULL;
  if (delayed_groups > 0) {
    std::vector<std::uint64_t> groups_to_remove;
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        if (!cpg.has_major_delay_groups_) {
          continue;
        }

        // remove groups without better alternatives from cpg
        // so that they are not included in the simulation
        // (they remain unchanged)
        utl::erase_if(cpg.groups_, [&](passenger_group const* pg) {
          if (pg_event_types.at(pg) !=
              monitoring_event_type::MAJOR_DELAY_EXPECTED) {
            return false;
          }
          auto const expected_current_arrival_time =
              expected_arrival_times.at(pg);
          utl::verify(expected_current_arrival_time != INVALID_TIME,
                      "invalid expected arrival time for delayed group");
          return !has_better_alternative(cpg.alternatives_,
                                         expected_current_arrival_time,
                                         min_delay_improvement_);
        });

        // groups with better alternatives are removed from the paxmon graph
        // and included in the simulation
        // temporary copy is needed because the original group is deleted
        for (auto& pg : cpg.groups_) {
          if (pg_event_types.at(pg) ==
              monitoring_event_type::MAJOR_DELAY_EXPECTED) {
            groups_to_remove.emplace_back(pg->id_);
            auto& copy = removed_groups.emplace_back(
                std::make_unique<passenger_group>(*pg));
            pg = copy.get();
            pg_event_types.insert(
                {pg, monitoring_event_type::MAJOR_DELAY_EXPECTED});
            ++tick_stats.major_delay_groups_with_alternatives_;
            ++removed_group_count;
          }
        }

        if (groups_to_remove.size() >= REMOVE_GROUPS_BATCH_SIZE) {
          send_remove_groups(uv, groups_to_remove, tick_stats);
        }
      }
    }
    send_remove_groups(uv, groups_to_remove, tick_stats);
    LOG(info) << "delayed groups: " << delayed_groups
              << ", removed groups: " << removed_group_count
              << " (tick total: " << tick_stats.removed_groups_ << ")";
  }

  MOTIS_START_TIMING(passenger_behavior);
  manual_timer sim_timer{"passenger behavior simulation"};
  auto pb = behavior::default_behavior{deterministic_mode_};
  auto const sim_result =
      simulate_behavior(sched, caps, uv, combined_groups, pb.pb_);
  sim_timer.stop_and_print();
  MOTIS_STOP_TIMING(passenger_behavior);
  tick_stats.t_passenger_behavior_ = MOTIS_TIMING_MS(passenger_behavior);

  LOG(info) << "forecast: " << sim_result.additional_groups_.size()
            << " edges affected";
  LOG(info) << fmt::format(
      "simulation average statistics: alternatives found: {:.2f}, alternatives "
      "picked: {:.2f}, P(best): {:.2f}%, P(2nd best): {:.2f}% ({} groups, {} "
      "combined)",
      sim_result.stats_.found_alt_count_avg_,
      sim_result.stats_.picked_alt_count_avg_,
      sim_result.stats_.best_alt_prob_avg_ * 100,
      sim_result.stats_.second_alt_prob_avg_ * 100,
      sim_result.stats_.group_count_, sim_result.stats_.combined_group_count_);

  if (behavior_stats_file_.is_open() && uv.id_ == 0) {
    fmt::print(behavior_stats_file_, "{},{},{},{:.4f},{:.4f},{:.2f},{:.2f}\n",
               static_cast<std::uint64_t>(sched.system_time_),
               sim_result.stats_.group_count_,
               sim_result.stats_.combined_group_count_,
               sim_result.stats_.found_alt_count_avg_,
               sim_result.stats_.picked_alt_count_avg_,
               sim_result.stats_.best_alt_prob_avg_ * 100,
               sim_result.stats_.second_alt_prob_avg_ * 100);
  }

  if (calc_load_forecast_) {
    MOTIS_START_TIMING(total_load_forecast);

    MOTIS_START_TIMING(calc_load_forecast);
    manual_timer load_forecast_timer{"load forecast"};
    auto const lfc = calc_load_forecast(sched, uv, sim_result);
    load_forecast_timer.stop_and_print();
    MOTIS_STOP_TIMING(calc_load_forecast);
    tick_stats.t_calc_load_forecast_ = MOTIS_TIMING_MS(calc_load_forecast);

    MOTIS_START_TIMING(load_forecast_fbs);
    manual_timer load_forecast_msg_timer{"load forecast make msg"};
    auto const forecast_msg =
        make_forecast_update_msg(sched, uv, sim_result, lfc);
    load_forecast_msg_timer.stop_and_print();
    MOTIS_STOP_TIMING(load_forecast_fbs);
    tick_stats.t_load_forecast_fbs_ = MOTIS_TIMING_MS(load_forecast_fbs);

    MOTIS_START_TIMING(write_load_forecast);
    if (forecast_file_.is_open() && uv.id_ == 0) {
      scoped_timer load_forecast_msg_timer{"load forecast to json"};
      forecast_file_ << forecast_msg->to_json(true) << std::endl;
    }
    MOTIS_STOP_TIMING(write_load_forecast);
    tick_stats.t_write_load_forecast_ = MOTIS_TIMING_MS(write_load_forecast);

    MOTIS_START_TIMING(publish_load_forecast);
    if (publish_load_forecast_) {
      ctx::await_all(motis_publish(forecast_msg));
    }
    MOTIS_STOP_TIMING(publish_load_forecast);
    tick_stats.t_publish_load_forecast_ =
        MOTIS_TIMING_MS(publish_load_forecast);

    MOTIS_STOP_TIMING(total_load_forecast);
    tick_stats.t_total_load_forecast_ = MOTIS_TIMING_MS(total_load_forecast);
  }

  MOTIS_START_TIMING(update_tracked_groups);
  scoped_timer update_tracked_groups_timer{"update tracked groups"};
  update_tracked_groups(sched, uv, sim_result, pg_event_types, tick_stats);
  MOTIS_STOP_TIMING(update_tracked_groups);
  tick_stats.t_update_tracked_groups_ = MOTIS_TIMING_MS(update_tracked_groups);

  MOTIS_STOP_TIMING(total);
  tick_stats.t_total_ = MOTIS_TIMING_MS(total);

  LOG(info) << "paxforecast tick stats: " << tick_stats.monitoring_events_
            << " monitoring events, " << tick_stats.groups_ << " groups ("
            << tick_stats.combined_groups_ << " combined), "
            << tick_stats.major_delay_groups_ << " major delay groups ("
            << tick_stats.major_delay_groups_with_alternatives_
            << " with alternatives), " << tick_stats.routing_requests_
            << " routing requests, " << tick_stats.alternatives_found_
            << " alternatives found, " << tick_stats.added_groups_
            << " groups added, " << tick_stats.removed_groups_
            << " groups removed";
  if (uv.id_ == 0) {
    stats_writer_->write_tick(tick_stats);
    stats_writer_->flush();
  }
}

msg_ptr paxforecast::apply_measures(msg_ptr const& msg) {
  scoped_timer all_timer{"apply_measures"};
  auto const req = motis_content(PaxForecastApplyMeasuresRequest, msg);
  auto& data =
      *get_shared_data<paxmon_data*>(to_res_id(global_res_id::PAX_DATA));
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto& caps = data.capacity_maps_;

  // update measures
  LOG(info) << "parse measures";
  auto const new_ms = from_fbs(sched, req->measures());
  LOG(info) << "get measure storage for universe " << req->universe();
  auto& measures = measures_storage_->get(req->universe());
  if (req->replace_existing()) {
    measures.clear();
  }
  LOG(info) << "storing measures";
  for (auto const& [t, nm] : new_ms) {
    auto& m = measures[t];
    m.reserve(m.size() + nm.size());
    std::copy(begin(nm), end(nm), std::back_inserter(m));
  }

  LOG(info) << "apply_measures: measures for " << measures.size()
            << " time points";

  uv.update_tracker_.start_tracking(uv, sched,
                                    req->include_before_trip_load_info(),
                                    req->include_after_trip_load_info());
  MOTIS_FINALLY([&]() { uv.update_tracker_.stop_tracking(); });

  // stats
  auto measure_time_points = 0ULL;
  auto total_measures_applied = 0ULL;
  auto total_affected_groups = 0ULL;
  auto total_alternative_routings = 0ULL;
  auto total_alternatives_found = 0ULL;
  // timings (ms)
  auto t_rt_updates = 0.;
  auto t_get_affected_groups = 0.;
  auto t_find_alternatives = 0.;
  auto t_add_alternatives_to_graph = 0.;
  auto t_behavior_simulation = 0.;
  auto t_update_groups = 0.;
  auto t_update_tracker = 0.;

  // simulate passenger behavior with measures
  for (auto const& [t, ms] : measures) {
    scoped_timer measure_timer{"measure"};
    ++measure_time_points;
    total_measures_applied += ms.size();
    auto const contains_rt_updates =
        std::any_of(begin(ms), end(ms), [](auto const& m) {
          return std::holds_alternative<measures::rt_update>(m);
        });

    LOG(info) << "apply_measures @" << format_time(t)
              << " [contains_rt_updates=" << contains_rt_updates
              << "], system_time=" << sched.system_time_
              << ", schedule_begin=" << sched.schedule_begin_;

    if (contains_rt_updates) {
      manual_timer rt_timer{"applying rt updates"};
      auto rt_lock =
          lock_resources({{uv.schedule_res_id_, ctx::access_t::WRITE}});
      message_creator mc;
      std::vector<flatbuffers::Offset<motis::ris::RISInputMessage>> rims;
      for (auto const& m : ms) {
        if (std::holds_alternative<measures::rt_update>(m)) {
          auto const rtum = std::get<measures::rt_update>(m);
          rims.emplace_back(CreateRISInputMessage(
              mc, rtum.type_, mc.CreateString(rtum.content_)));
        }
      }
      // TODO(pablo): check for errors? -> ri basis parser should throw errors
      mc.create_and_finish(
          MsgContent_RISApplyRequest,
          CreateRISApplyRequest(mc, uv.schedule_res_id_, mc.CreateVector(rims))
              .Union(),
          "/ris/apply");
      motis_call(make_msg(mc))->val();
      rt_timer.stop_and_print();
      t_rt_updates += rt_timer.duration_ms();
    }

    auto const loc_time = t + req->preparation_time();
    manual_timer get_affected_groups_timer{"get_affected_grous"};
    auto const affected_groups =
        measures::get_affected_groups(sched, uv, loc_time, ms);
    get_affected_groups_timer.stop_and_print();
    t_get_affected_groups += get_affected_groups_timer.duration_ms();

    total_affected_groups += affected_groups.measures_.size();
    LOG(info) << "affected groups: " << affected_groups.measures_.size();

    // combine groups by (localization, remaining planned journey)
    auto combined =
        mcd::hash_map<mcd::pair<passenger_localization, compact_journey>,
                      combined_passenger_group>{};
    for (auto const& [pg, loc] : affected_groups.localization_) {
      auto const remaining_planned_journey =
          get_suffix(sched, pg->compact_planned_journey_, loc);
      if (remaining_planned_journey.legs_.empty()) {
        continue;
      }
      auto& cpg = combined[{loc, remaining_planned_journey}];
      cpg.groups_.emplace_back(pg);
      cpg.passengers_ += pg->passengers_;
      cpg.localization_ = loc;
    }

    LOG(info) << "combined: " << combined.size();

    manual_timer alternatives_timer{"apply_measures: find alternatives"};
    std::vector<ctx::future_ptr<ctx_data, void>> futures;
    for (auto& ce : combined) {
      auto const& loc = ce.first.first;
      auto const& remaining_planned_journey = ce.first.second;
      auto& cpg = ce.second;

      std::set<measures::measure_variant const*> measures_set;
      for (auto const* pg : cpg.groups_) {
        for (auto const* mv : affected_groups.measures_.at(pg)) {
          measures_set.insert(mv);
        }
      }
      auto group_measures = mcd::to_vec(measures_set);
      futures.emplace_back(spawn_job_void([&, this, group_measures] {
        cpg.alternatives_ = find_alternatives(
            uv, sched, routing_cache_, group_measures,
            remaining_planned_journey.destination_station_id(), loc,
            &remaining_planned_journey, false, 61);
      }));
    }
    ctx::await_all(futures);
    routing_cache_.sync();
    alternatives_timer.stop_and_print();
    t_find_alternatives += alternatives_timer.duration_ms();
    total_alternative_routings += combined.size();

    {
      manual_timer alt_trips_timer{"add alternatives to graph"};
      for (auto& [grp_key, cpg] : combined) {
        total_alternatives_found += cpg.alternatives_.size();
        for (auto const& alt : cpg.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, caps, uv, leg.trip_idx_);
          }
        }
      }
      alt_trips_timer.stop_and_print();
      t_add_alternatives_to_graph += alt_trips_timer.duration_ms();
    }

    manual_timer sim_timer{"passenger behavior simulation"};
    auto pb = behavior::default_behavior{deterministic_mode_};
    auto const sim_result =
        simulate_behavior(sched, caps, uv, combined, pb.pb_);
    sim_timer.stop_and_print();
    t_behavior_simulation += sim_timer.duration_ms();

    manual_timer update_groups_timer{"update groups"};
    tick_statistics tick_stats;
    update_tracked_groups(sched, uv, sim_result, {}, tick_stats);
    update_groups_timer.stop_and_print();
    t_update_groups += update_groups_timer.duration_ms();
  }

  manual_timer update_tracker_timer{"update tracker"};
  auto [mc, fb_updates] = uv.update_tracker_.finish_updates();
  update_tracker_timer.stop_and_print();
  t_update_tracker = update_tracker_timer.duration_ms();

  mc.create_and_finish(
      MsgContent_PaxForecastApplyMeasuresResponse,
      CreatePaxForecastApplyMeasuresResponse(
          mc,
          CreatePaxForecastApplyMeasuresStatistics(
              mc, measure_time_points, total_measures_applied,
              total_affected_groups, total_alternative_routings,
              total_alternatives_found, t_rt_updates, t_get_affected_groups,
              t_find_alternatives, t_add_alternatives_to_graph,
              t_behavior_simulation, t_update_groups, t_update_tracker),
          fb_updates)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxforecast
