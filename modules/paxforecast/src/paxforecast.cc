#include "motis/paxforecast/paxforecast.h"

#include <ctime>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>

#include "fmt/format.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/service_access.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/message.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/data_key.h"
#include "motis/paxmon/debug.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/paxmon_data.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/load_forecast.h"
#include "motis/paxforecast/measures/measures.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/statistics.h"

#include "motis/paxforecast/behavior/logit/conditional_logit_passenger_behavior.h"
#include "motis/paxforecast/behavior/logit/mixed_logit_passenger_behavior.h"
#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;
using namespace motis::paxmon;

namespace motis::paxforecast {

paxforecast::paxforecast() : module("Passenger Forecast", "paxforecast") {
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

  reg.subscribe("/paxmon/monitoring_update", [&](msg_ptr const& msg) {
    on_monitoring_event(msg);
    return nullptr;
  });
}

auto const constexpr REMOVE_GROUPS_BATCH_SIZE = 10'000;
auto const constexpr ADD_GROUPS_BATCH_SIZE = 10'000;

void send_remove_groups(std::vector<std::uint64_t>& groups_to_remove,
                        tick_statistics& tick_stats) {
  if (groups_to_remove.empty()) {
    return;
  }
  tick_stats.removed_groups_ += groups_to_remove.size();
  message_creator remove_groups_mc;
  remove_groups_mc.create_and_finish(
      MsgContent_PaxMonRemoveGroupsRequest,
      CreatePaxMonRemoveGroupsRequest(
          remove_groups_mc, remove_groups_mc.CreateVector(groups_to_remove))
          .Union(),
      "/paxmon/remove_groups");
  auto const remove_msg = make_msg(remove_groups_mc);
  motis_call(remove_msg)->val();
  groups_to_remove.clear();
}

void update_tracked_groups(
    schedule const& sched, simulation_result const& sim_result,
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
        CreatePaxMonAddGroupsRequest(add_groups_mc,
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

    auto const event_type = pg_event_types.at(pg);
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
      send_remove_groups(groups_to_remove, tick_stats);
    }
    if (groups_to_add.size() >= ADD_GROUPS_BATCH_SIZE) {
      send_add_groups();
    }
  }

  LOG(info) << "update_tracked_groups: -" << remove_group_count << " +"
            << add_group_count;
  tick_stats.added_groups_ += add_group_count;

  send_remove_groups(groups_to_remove, tick_stats);
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
  tick_statistics tick_stats;
  MOTIS_START_TIMING(total);
  auto const& sched = get_sched();
  tick_stats.system_time_ = sched.system_time_;
  auto& data = *get_shared_data<paxmon_data*>(motis::paxmon::DATA_KEY);
  auto& caps = *get_shared_data<capacity_maps*>(motis::paxmon::CAPS_KEY);

  auto const mon_update = motis_content(PaxMonUpdate, msg);

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  std::map<unsigned, std::vector<combined_passenger_group>> combined_groups;
  std::map<passenger_group const*, monitoring_event_type> pg_event_types;
  std::map<passenger_group const*, time> expected_arrival_times;
  auto delayed_groups = 0ULL;

  for (auto const& event : *mon_update->events()) {
    if (event->type() == PaxMonEventType_NO_PROBLEM) {
      continue;
    }

    auto const pg = data.get_passenger_group(event->group()->id());
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
    scoped_timer alt_timer{"find alternatives"};
    std::vector<ctx::future_ptr<ctx_data, void>> futures;
    for (auto& cgs : combined_groups) {
      auto const destination_station_id = cgs.first;
      for (auto& cpg : cgs.second) {
        ++routing_requests;
        futures.emplace_back(spawn_job_void([this, &sched,
                                             destination_station_id, &cpg] {
          cpg.alternatives_ = find_alternatives(
              sched, destination_station_id, cpg.localization_, routing_cache_);
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
            get_or_add_trip(sched, caps, data, leg.trip_);
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
          send_remove_groups(groups_to_remove, tick_stats);
        }
      }
    }
    send_remove_groups(groups_to_remove, tick_stats);
    LOG(info) << "delayed groups: " << delayed_groups
              << ", removed groups: " << removed_group_count
              << " (tick total: " << tick_stats.removed_groups_ << ")";
  }

  MOTIS_START_TIMING(passenger_behavior);
  manual_timer sim_timer{"passenger behavior simulation"};
#ifdef WIN32
  auto const seed = static_cast<std::mt19937::result_type>(
      std::time(nullptr) %
      std::numeric_limits<std::mt19937::result_type>::max());
#else
  auto rd = std::random_device();
  auto const seed = rd();
#endif
  auto rnd_gen = std::mt19937{seed};

  auto transfer_dist = std::normal_distribution<float>{30.0F, 10.0F};
  auto pb = behavior::probabilistic::passenger_behavior{
      rnd_gen, transfer_dist, 1000, deterministic_mode_};

  /*
  // TrSimple / TrainSimple.ml
  auto pb = behavior::logit::conditional_logit_passenger_behavior{
      0.0011500F, -0.0973259F, deterministic_mode_};
  */

  /*
  // TrMin / TrainMin.ml
  auto pb = behavior::logit::conditional_logit_passenger_behavior{
      2.8676e-02, 3.2634e-01, deterministic_mode_};
  */

  /*
  // TrSimple / TrainSimple.mxlu
  auto transfer_dist = std::normal_distribution<float>{-0.1040768F, 0.3331784F};
  auto pb = behavior::logit::mixed_logit_passenger_behavior{
      rnd_gen, 0.0012139F, transfer_dist, 1000, deterministic_mode_};
  */

  /*
  // TrMin / TrainMin.mxlu
  auto transfer_dist =
      std::normal_distribution<float>{3.2634e-01F, 1.0000e-01F};
  auto pb = behavior::logit::mixed_logit_passenger_behavior{
      rnd_gen, 2.8676e-02F, transfer_dist, 1000, deterministic_mode_};
  */

  auto const announcements = std::vector<measures::please_use>{};
  auto const sim_result =
      simulate_behavior(sched, caps, data, combined_groups, announcements, pb);
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

  if (behavior_stats_file_.is_open()) {
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
    auto const lfc = calc_load_forecast(sched, data, sim_result);
    load_forecast_timer.stop_and_print();
    MOTIS_STOP_TIMING(calc_load_forecast);
    tick_stats.t_calc_load_forecast_ = MOTIS_TIMING_MS(calc_load_forecast);

    MOTIS_START_TIMING(load_forecast_fbs);
    manual_timer load_forecast_msg_timer{"load forecast make msg"};
    auto const forecast_msg =
        make_forecast_update_msg(sched, data, sim_result, lfc);
    load_forecast_msg_timer.stop_and_print();
    MOTIS_STOP_TIMING(load_forecast_fbs);
    tick_stats.t_load_forecast_fbs_ = MOTIS_TIMING_MS(load_forecast_fbs);

    MOTIS_START_TIMING(write_load_forecast);
    if (forecast_file_.is_open()) {
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
  update_tracked_groups(sched, sim_result, pg_event_types, tick_stats);
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
  ;
  stats_writer_->write_tick(tick_stats);
  stats_writer_->flush();
}

}  // namespace motis::paxforecast
