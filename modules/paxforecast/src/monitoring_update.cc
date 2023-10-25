#include "motis/paxforecast/monitoring_update.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <vector>

#include "fmt/format.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/core/access/trip_access.h"
#include "motis/core/debug/trip.h"
#include "motis/paxmon/debug.h"

#include "motis/paxmon/fbs_compact_journey_util.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/localization_conv.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"

#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/paxforecast.h"
#include "motis/paxforecast/revert_forecast.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/universe_data.h"
#include "motis/paxforecast/update_tracked_groups.h"

#include "motis/paxforecast/behavior/default_behavior.h"

using namespace motis::paxmon;
using namespace motis::module;
using namespace motis::logging;

namespace motis::paxforecast {

auto const constexpr REMOVE_GROUPS_BATCH_SIZE = 10'000;

struct passenger_group_with_route_and_localization {
  passenger_group_with_route pgwr_{};
  passenger_localization const* loc_{};
};

using combined_groups_map_t = std::map<unsigned /* destination station id */,
                                       std::vector<combined_passenger_group>>;

void log_destination_reachable(
    universe& uv, schedule const& sched,
    passenger_group_with_route_and_probability const& pgwrap,
    passenger_localization const& loc) {
  auto& route = uv.passenger_groups_.route(pgwrap.pgwr_);
  route.destination_unreachable_ = false;
  if (pgwrap.probability_ == 0.0F) {
    return;
  }
  auto log_entries = uv.passenger_groups_.reroute_log_entries(pgwrap.pgwr_.pg_);
  if (auto it = std::find_if(log_entries.rbegin(), log_entries.rend(),
                             [&](reroute_log_entry const& entry) {
                               return entry.old_route_.route_ ==
                                      pgwrap.pgwr_.route_;
                             });
      it != log_entries.rend()) {
    if (it->reason_ != reroute_reason_t::DESTINATION_UNREACHABLE) {
      return;
    }
    auto const log_new_routes =
        uv.passenger_groups_.log_entry_new_routes_.emplace_back();
    log_entries.emplace_back(reroute_log_entry{
        static_cast<reroute_log_entry_index>(log_new_routes.index()),
        reroute_log_route_info{pgwrap.pgwr_.route_, pgwrap.probability_,
                               pgwrap.probability_, to_log_localization(loc)},
        sched.system_time_,
        now(),
        uv.update_number_,
        reroute_reason_t::DESTINATION_REACHABLE,
        {}});
  }
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

void send_remove_group_routes(
    schedule const& sched, universe const& uv,
    std::vector<passenger_group_with_route_and_localization>&
        group_routes_to_remove,
    tick_statistics& tick_stats, reroute_reason_t const reason) {
  if (group_routes_to_remove.empty()) {
    return;
  }
  LOG(info) << "removing " << group_routes_to_remove.size() << " group routes";
  tick_stats.removed_group_routes_ += group_routes_to_remove.size();
  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonRerouteGroupsRequest,
      CreatePaxMonRerouteGroupsRequest(
          mc, uv.id_,
          mc.CreateVector(utl::to_vec(
              group_routes_to_remove,
              [&](auto const& pgwrl) {
                auto const& pgwr = pgwrl.pgwr_;
                return CreatePaxMonRerouteGroup(
                    mc, pgwr.pg_, pgwr.route_,
                    mc.CreateVector(
                        std::vector<flatbuffers::Offset<PaxMonGroupRoute>>{}),
                    static_cast<PaxMonRerouteReason>(reason),
                    broken_transfer_info_to_fbs(mc, sched, {}), false,
                    mc.CreateVector(std::vector{
                        to_fbs_localization_wrapper(sched, mc, *pgwrl.loc_)}));
              })))
          .Union(),
      "/paxmon/reroute_groups");
  auto const remove_msg = make_msg(mc);
  motis_call(remove_msg)->val();
  group_routes_to_remove.clear();
}

void find_cpg_alternatives(paxforecast& mod, universe& uv,
                           schedule const& sched, tick_statistics& tick_stats,
                           combined_groups_map_t& combined_groups) {
  auto routing_requests = 0ULL;
  auto alternatives_found = 0ULL;

  {
    MOTIS_START_TIMING(find_alternatives);
    scoped_timer const alt_timer{"on_monitoring_event: find alternatives"};
    std::vector<ctx::future_ptr<ctx_data, void>> futures;
    for (auto& cgs : combined_groups) {
      auto const destination_station_id = cgs.first;
      for (auto& cpg : cgs.second) {
        ++routing_requests;
        futures.emplace_back(
            spawn_job_void([&mod, &uv, &sched, destination_station_id, &cpg] {
              cpg.alternatives_ = find_alternatives(
                  uv, sched, mod.routing_cache_, {}, destination_station_id,
                  cpg.localization_, nullptr, true, 0, mod.allow_start_metas_,
                  mod.allow_dest_metas_);
            }));
      }
    }
    LOG(info) << "find alternatives: " << routing_requests
              << " routing requests (using cache="
              << mod.routing_cache_.is_open() << ")...";
    tick_stats.routing_requests_ += routing_requests;
    ctx::await_all(futures);
    mod.routing_cache_.sync();
    MOTIS_STOP_TIMING(find_alternatives);
    tick_stats.t_find_alternatives_ += MOTIS_TIMING_MS(find_alternatives);
  }

  {
    MOTIS_START_TIMING(add_alternatives);
    scoped_timer const alt_trips_timer{"add alternatives to graph"};
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        alternatives_found += cpg.alternatives_.size();
        for (auto const& alt : cpg.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, uv, leg.trip_idx_);
          }
        }
      }
    }
    tick_stats.alternatives_found_ += alternatives_found;
    MOTIS_STOP_TIMING(add_alternatives);
    tick_stats.t_add_alternatives_ += MOTIS_TIMING_MS(add_alternatives);
  }

  LOG(info) << "alternatives: " << routing_requests << " routing requests => "
            << alternatives_found << " alternatives";
}

passenger_group_with_route_and_probability event_to_pgwrap(
    PaxMonEvent const* event) {
  return passenger_group_with_route_and_probability{
      passenger_group_with_route{
          static_cast<passenger_group_index>(event->group_route()->group_id()),
          static_cast<local_group_route_index>(
              event->group_route()->route()->index())},
      event->group_route()->route()->probability(),
      event->group_route()->passenger_count()};
}

void add_to_cpg(combined_groups_map_t& combined_groups,
                passenger_group_with_route_and_probability const& pgwrap,
                passenger_localization const& localization,
                unsigned const destination_station_id,
                std::uint16_t const passenger_count) {
  auto& destination_groups = combined_groups[destination_station_id];
  // TODO(pablo): localization includes the scheduled arrival time, which
  //  is needed later (journey prefix calculation). to make sure this works,
  //  the scheduled time is currently included in the comparison.
  //  it might be better to only check the current arrival time
  //  and store the scheduled arrival time / localization per group
  //  instead of per combined group.
  auto cpg = std::find_if(
      begin(destination_groups), end(destination_groups),
      [&](auto const& g) { return g.localization_ == localization; });
  if (cpg == end(destination_groups)) {
    destination_groups.emplace_back(combined_passenger_group{
        destination_station_id, passenger_count, localization, {pgwrap}, {}});
  } else {
    cpg->passengers_ += passenger_count;
    cpg->group_routes_.push_back(pgwrap);
  }
}

mcd::hash_map<passenger_group_with_route, passenger_localization const*>
get_localization_refs(combined_groups_map_t const& combined_groups) {
  auto pgwr_localizations = mcd::hash_map<passenger_group_with_route,
                                          passenger_localization const*>{};
  for (auto& cgs : combined_groups) {
    for (auto& cpg : cgs.second) {
      for (auto const& pgwrap : cpg.group_routes_) {
        pgwr_localizations[pgwrap.pgwr_] = &cpg.localization_;
      }
    }
  }

  return pgwr_localizations;
}

simulation_result run_simulation(paxforecast& mod, universe& uv,
                                 schedule const& sched,
                                 tick_statistics& tick_stats,
                                 combined_groups_map_t const& combined_groups,
                                 char const* event_name) {
  MOTIS_START_TIMING(passenger_behavior);
  manual_timer sim_timer{"passenger behavior simulation (broken transfers)"};
  auto pb = behavior::default_behavior{mod.deterministic_mode_};
  auto const sim_result = simulate_behavior(sched, uv, combined_groups, pb.pb_,
                                            mod.probability_threshold_);
  sim_timer.stop_and_print();
  MOTIS_STOP_TIMING(passenger_behavior);
  tick_stats.t_passenger_behavior_ += MOTIS_TIMING_MS(passenger_behavior);

  LOG(info) << fmt::format(
      "forecast[{}]: {} edges affected, simulation average statistics: "
      "alternatives found: {:.2f}, alternatives "
      "picked: {:.2f}, P(best): {:.2f}%, P(2nd best): {:.2f}% ({} group "
      "routes, {} "
      "combined) (broken transfers)",
      event_name, sim_result.additional_groups_.size(),
      sim_result.stats_.found_alt_count_avg_,
      sim_result.stats_.picked_alt_count_avg_,
      sim_result.stats_.best_alt_prob_avg_ * 100,
      sim_result.stats_.second_alt_prob_avg_ * 100,
      sim_result.stats_.group_route_count_,
      sim_result.stats_.combined_group_count_);

  if (mod.behavior_stats_file_.is_open() && uv.id_ == 0) {
    fmt::print(mod.behavior_stats_file_,
               "{},{},{},{},{:.4f},{:.4f},{:.2f},{:.2f}\n",
               static_cast<std::uint64_t>(sched.system_time_), event_name,
               sim_result.stats_.group_route_count_,
               sim_result.stats_.combined_group_count_,
               sim_result.stats_.found_alt_count_avg_,
               sim_result.stats_.picked_alt_count_avg_,
               sim_result.stats_.best_alt_prob_avg_ * 100,
               sim_result.stats_.second_alt_prob_avg_ * 100);
  }

  // TODO(pablo): calc_load_forecast if mode.calc_load_forecast_

  return sim_result;
}

void update_groups(
    universe& uv, schedule const& sched, simulation_result const& sim_result,
    mcd::hash_map<passenger_group_with_route,
                  passenger_localization const*> const& pgwr_localizations,
    std::map<passenger_group_with_route,
             std::optional<broken_transfer_info>> const& broken_transfer_infos,
    tick_statistics& tick_stats) {
  MOTIS_START_TIMING(update_tracked_groups);
  scoped_timer const update_tracked_groups_timer{"update tracked groups"};
  update_tracked_groups(sched, uv, sim_result, broken_transfer_infos,
                        pgwr_localizations, tick_stats,
                        reroute_reason_t::BROKEN_TRANSFER);
  MOTIS_STOP_TIMING(update_tracked_groups);
  tick_stats.t_update_tracked_groups_ += MOTIS_TIMING_MS(update_tracked_groups);
}

void handle_broken_transfers(paxforecast& mod, universe& uv,
                             schedule const& sched, tick_statistics& tick_stats,
                             PaxMonUpdate const* mon_update) {
  auto broken_transfer_infos = std::map<passenger_group_with_route,
                                        std::optional<broken_transfer_info>>{};
  auto combined_groups = std::map<unsigned /* destination station id */,
                                  std::vector<combined_passenger_group>>{};

  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_BROKEN_TRANSFER) {
      continue;
    }

    auto const pgwrap = event_to_pgwrap(event);
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id = get_destination_station_id(
        sched, event->group_route()->route()->journey());
    auto const next_stop_is_destination =
        localization.at_station_->index_ == destination_station_id;

    // <debug>
    if (next_stop_is_destination) {
      std::cout << "[MU]: next stop is destination: pg=" << pgwrap.pgwr_.pg_
                << "#" << pgwrap.pgwr_.route_
                << ", event=" << static_cast<int>(event->type())
                << ", prob=" << pgwrap.probability_
                << ", station=" << localization.at_station_->eva_nr_ << " ("
                << localization.at_station_->name_
                << "), eta=" << format_unix_time(event->expected_arrival_time())
                << ", planned="
                << format_unix_time(
                       event->group_route()->route()->planned_arrival_time())
                << ", delay="
                << event->group_route()->route()->estimated_delay()
                << ", broken=" << event->group_route()->route()->broken()
                << ", in_trip=" << localization.in_trip() << std::endl;
    }
    // </debug>

    if (next_stop_is_destination || pgwrap.probability_ == 0.F) {
      continue;
    }

    broken_transfer_infos[pgwrap.pgwr_] =
        from_fbs(sched, event->reachability()->broken_transfer());

    add_to_cpg(combined_groups, pgwrap, localization, destination_station_id,
               event->group_route()->passenger_count());
  }

  if (combined_groups.empty()) {
    return;
  }

  auto const pgwr_localizations = get_localization_refs(combined_groups);

  tick_stats.group_routes_ += pgwr_localizations.size();
  tick_stats.combined_groups_ += combined_groups.size();

  find_cpg_alternatives(mod, uv, sched, tick_stats, combined_groups);

  auto const sim_result =
      run_simulation(mod, uv, sched, tick_stats, combined_groups, "broken");

  update_groups(uv, sched, sim_result, pgwr_localizations,
                broken_transfer_infos, tick_stats);
}

void handle_major_delays(paxforecast& mod, universe& uv, schedule const& sched,
                         tick_statistics& tick_stats,
                         PaxMonUpdate const* mon_update) {
  auto combined_groups = std::map<unsigned /* destination station id */,
                                  std::vector<combined_passenger_group>>{};
  auto expected_arrival_times = std::map<passenger_group_with_route, time>{};

  auto last_pgi = std::numeric_limits<passenger_group_index>::max();
  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_MAJOR_DELAY_EXPECTED) {
      continue;
    }

    auto pgwrap = event_to_pgwrap(event);
    // probability may have changed because of broken transfers in other
    // routes of the same group
    pgwrap.probability_ = uv.passenger_groups_.route(pgwrap.pgwr_).probability_;
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id = get_destination_station_id(
        sched, event->group_route()->route()->journey());

    if (pgwrap.probability_ == 0.F) {
      continue;
    }

    expected_arrival_times.insert(
        {pgwrap.pgwr_, unix_to_motistime(sched.schedule_begin_,
                                         event->expected_arrival_time())});

    if (last_pgi == pgwrap.pgwr_.pg_) {
      // TODO(pablo): major delays for more than one route per group
      //  are currently not supported because processing them separately
      //  breaks things
      LOG(info) << "skipping major delay for group " << pgwrap.pgwr_.pg_
                << " because of multiple delayed routes";
      continue;
    }
    last_pgi = pgwrap.pgwr_.pg_;

    add_to_cpg(combined_groups, pgwrap, localization, destination_station_id,
               event->group_route()->passenger_count());
  }

  if (combined_groups.empty()) {
    return;
  }

  LOG(info) << "handle_major_delays: " << expected_arrival_times.size()
            << " groups, " << combined_groups.size() << " combined groups";

  auto const pgwr_localizations = get_localization_refs(combined_groups);

  tick_stats.group_routes_ += pgwr_localizations.size();
  tick_stats.combined_groups_ += combined_groups.size();
  tick_stats.major_delay_group_routes_ += expected_arrival_times.size();

  find_cpg_alternatives(mod, uv, sched, tick_stats, combined_groups);

  auto removed_group_route_count = 0ULL;
  auto group_routes_to_remove =
      std::vector<passenger_group_with_route_and_localization>{};
  for (auto& cgs : combined_groups) {
    for (auto& cpg : cgs.second) {
      // remove groups without better alternatives from cpg
      // so that they are not included in the simulation
      // (they remain unchanged)
      utl::erase_if(
          cpg.group_routes_,
          [&](passenger_group_with_route_and_probability const& pgwrap) {
            auto const expected_current_arrival_time =
                expected_arrival_times.at(pgwrap.pgwr_);
            utl::verify(expected_current_arrival_time != INVALID_TIME,
                        "invalid expected arrival time for delayed group");
            return !has_better_alternative(cpg.alternatives_,
                                           expected_current_arrival_time,
                                           mod.min_delay_improvement_);
          });

      // group routes with better alternatives are removed from the paxmon
      // graph and included in the simulation
      for (auto const& pgwrap : cpg.group_routes_) {
        group_routes_to_remove.emplace_back(
            passenger_group_with_route_and_localization{pgwrap.pgwr_,
                                                        &cpg.localization_});
        ++tick_stats.major_delay_group_routes_with_alternatives_;
        ++removed_group_route_count;
      }

      if (group_routes_to_remove.size() >= REMOVE_GROUPS_BATCH_SIZE) {
        send_remove_group_routes(sched, uv, group_routes_to_remove, tick_stats,
                                 reroute_reason_t::MAJOR_DELAY_EXPECTED);
      }
    }
  }
  send_remove_group_routes(sched, uv, group_routes_to_remove, tick_stats,
                           reroute_reason_t::MAJOR_DELAY_EXPECTED);
  LOG(info) << "delayed group routes: " << tick_stats.major_delay_group_routes_
            << ", removed group routes: " << removed_group_route_count
            << " (tick total: " << tick_stats.removed_group_routes_ << ")";

  auto const sim_result =
      run_simulation(mod, uv, sched, tick_stats, combined_groups, "delay");

  update_groups(uv, sched, sim_result, pgwr_localizations, {}, tick_stats);
}

void handle_unbroken_transfers(paxforecast& mod, universe& uv,
                               schedule const& sched,
                               tick_statistics& tick_stats,
                               PaxMonUpdate const* mon_update) {
  auto unbroken_transfers = std::vector<passenger_group_with_route>{};

  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_NO_PROBLEM) {
      continue;
    }

    auto const pgwr = passenger_group_with_route{
        static_cast<passenger_group_index>(event->group_route()->group_id()),
        static_cast<local_group_route_index>(
            event->group_route()->route()->index())};
    auto pgwrap = passenger_group_with_route_and_probability{
        pgwr, event->group_route()->route()->probability(),
        event->group_route()->passenger_count()};
    // probability may have changed because of broken transfers in other
    // routes of the same group
    pgwrap.probability_ = uv.passenger_groups_.route(pgwr).probability_;
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());

    log_destination_reachable(uv, sched, pgwrap, localization);
    unbroken_transfers.push_back(pgwr);
  }

  if (!unbroken_transfers.empty() && mod.revert_forecasts_) {
    revert_forecasts(uv, sched, unbroken_transfers);
  }
}

void on_monitoring_update(paxforecast& mod, paxmon_data& data,
                          msg_ptr const& msg) {
  auto const mon_update = motis_content(PaxMonUpdate, msg);
  MOTIS_START_TIMING(total);
  auto const uv_access = get_universe_and_schedule(data, mon_update->universe(),
                                                   ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  auto tick_stats = tick_statistics{
      .system_time_ = static_cast<std::uint64_t>(sched.system_time_),
      .monitoring_events_ = mon_update->events()->size()};

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME,
              "paxforecast::on_monitoring_event: invalid current system time: "
              "system_time={}, schedule_begin={}",
              sched.system_time_, sched.schedule_begin_);

  handle_broken_transfers(mod, uv, sched, tick_stats, mon_update);
  handle_major_delays(mod, uv, sched, tick_stats, mon_update);
  handle_unbroken_transfers(mod, uv, sched, tick_stats, mon_update);

  MOTIS_STOP_TIMING(total);
  tick_stats.t_total_ = MOTIS_TIMING_MS(total);

  LOG(info) << "paxforecast tick stats: " << tick_stats.monitoring_events_
            << " monitoring events, " << tick_stats.group_routes_
            << " group routes (" << tick_stats.combined_groups_
            << " combined), " << tick_stats.major_delay_group_routes_
            << " major delay group routes ("
            << tick_stats.major_delay_group_routes_with_alternatives_
            << " with alternatives), " << tick_stats.routing_requests_
            << " routing requests, " << tick_stats.alternatives_found_
            << " alternatives found, " << tick_stats.rerouted_group_routes_
            << " group routes rerouted, " << tick_stats.removed_group_routes_
            << " group routes removed";
  if (uv.id_ == 0) {
    mod.stats_writer_->write_tick(tick_stats);
    mod.stats_writer_->flush();
  }

  auto& metrics = mod.universe_storage_.get(uv.id_).metrics_;
  metrics.add(sched.system_time_, now(), tick_stats);
}

}  // namespace motis::paxforecast
