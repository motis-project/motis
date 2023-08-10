#include "motis/paxforecast/monitoring_update.h"

#include <map>
#include <optional>
#include <vector>

#include "fmt/format.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"

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
                               pgwrap.probability_},
        sched.system_time_,
        now(),
        reroute_reason_t::DESTINATION_REACHABLE,
        {},
        to_log_localization(loc)});
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

// TODO(pablo): major delay groups -> "broken_group_routes"
void send_remove_group_routes(
    schedule const& sched, universe const& uv,
    std::vector<passenger_group_with_route_and_localization>&
        group_routes_to_remove,
    std::map<passenger_group_with_route,
             std::optional<broken_transfer_info>> const& broken_transfer_infos,
    tick_statistics& tick_stats, reroute_reason_t const reason) {
  if (group_routes_to_remove.empty()) {
    return;
  }
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
                    broken_transfer_info_to_fbs(mc, sched,
                                                broken_transfer_infos.at(pgwr)),
                    false,
                    mc.CreateVector(std::vector{
                        to_fbs_localization_wrapper(sched, mc, *pgwrl.loc_)}));
              })))
          .Union(),
      "/paxmon/reroute_groups");
  auto const remove_msg = make_msg(mc);
  motis_call(remove_msg)->val();
  group_routes_to_remove.clear();
}

void on_monitoring_update(paxforecast& mod, paxmon_data& data,
                          msg_ptr const& msg) {
  auto const mon_update = motis_content(PaxMonUpdate, msg);
  MOTIS_START_TIMING(total);
  auto const uv_access = get_universe_and_schedule(data, mon_update->universe(),
                                                   ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  tick_statistics tick_stats;
  tick_stats.system_time_ = sched.system_time_;

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME,
              "paxforecast::on_monitoring_event: invalid current system time: "
              "system_time={}, schedule_begin={}",
              sched.system_time_, sched.schedule_begin_);

  std::map<unsigned /* destination station id */,
           std::vector<combined_passenger_group>>
      combined_groups;
  std::map<passenger_group_with_route, monitoring_event_type> pgwr_event_types;
  std::map<passenger_group_with_route, time> expected_arrival_times;
  std::map<passenger_group_with_route, std::optional<broken_transfer_info>>
      broken_transfer_infos;
  std::vector<passenger_group_with_route> unbroken_transfers;
  auto delayed_group_routes = 0ULL;

  for (auto const& event : *mon_update->events()) {
    auto const pgwr = passenger_group_with_route{
        static_cast<passenger_group_index>(event->group_route()->group_id()),
        static_cast<local_group_route_index>(
            event->group_route()->route()->index())};
    auto pgwrap = passenger_group_with_route_and_probability{
        pgwr, event->group_route()->route()->probability(),
        event->group_route()->passenger_count()};
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id = get_destination_station_id(
        sched, event->group_route()->route()->journey());

    auto const next_stop_is_destination =
        localization.at_station_->index_ == destination_station_id;

    if (event->type() == PaxMonEventType_NO_PROBLEM) {
      unbroken_transfers.push_back(pgwr);
      log_destination_reachable(uv, sched, pgwrap, localization);
      // TODO(pablo): if current p=0, behavior simulation won't work
      // check if we need it anyway
      continue;
      // if (event->group_route()->route()->planned()) {
      //   continue;
      // }
    } else if ((next_stop_is_destination &&
                event->type() != PaxMonEventType_BROKEN_TRANSFER) ||
               pgwrap.probability_ == 0.0F) {
      continue;
    }

    auto const major_delay =
        event->type() == PaxMonEventType_MAJOR_DELAY_EXPECTED;

    if (major_delay) {
      ++delayed_group_routes;
      expected_arrival_times.insert(
          {pgwr, unix_to_motistime(sched.schedule_begin_,
                                   event->expected_arrival_time())});
    }

    auto const inserted = pgwr_event_types.insert(
        {pgwr, static_cast<monitoring_event_type>(event->type())});
    utl::verify(inserted.second,
                "multiple monitoring updates for passenger group");
    broken_transfer_infos[pgwr] =
        from_fbs(sched, event->reachability()->broken_transfer());

    auto& destination_groups = combined_groups[destination_station_id];
    // TODO(pablo): localization includes the scheduled arrival time, which
    // is needed later (journey prefix calculation). to make sure this works,
    // the scheduled time is currently included in the comparison.
    // it might be better to only check the current arrival time
    // and store the scheduled arrival time / localization per group
    // instead of per combined group.
    auto cpg = std::find_if(
        begin(destination_groups), end(destination_groups),
        [&](auto const& g) { return g.localization_ == localization; });
    if (cpg == end(destination_groups)) {
      destination_groups.emplace_back(
          combined_passenger_group{destination_station_id,
                                   event->group_route()->passenger_count(),
                                   major_delay,
                                   localization,
                                   {pgwrap},
                                   {}});
    } else {
      cpg->passengers_ += event->group_route()->passenger_count();
      cpg->group_routes_.push_back(pgwrap);
      if (major_delay) {
        cpg->has_major_delay_groups_ = true;
      }
    }
  }

  mcd::hash_map<passenger_group_with_route, passenger_localization const*>
      pgwr_localizations;
  for (auto& cgs : combined_groups) {
    for (auto& cpg : cgs.second) {
      for (auto const& pgwrap : cpg.group_routes_) {
        pgwr_localizations[pgwrap.pgwr_] = &cpg.localization_;
      }
    }
  }

  if (combined_groups.empty()) {
    if (!unbroken_transfers.empty() && mod.revert_forecasts_) {
      revert_forecasts(uv, sched, simulation_result{}, unbroken_transfers,
                       pgwr_localizations);
    }
    return;
  }

  LOG(info) << mon_update->events()->size() << " monitoring updates, "
            << pgwr_event_types.size() << " groups, " << combined_groups.size()
            << " combined groups";

  tick_stats.monitoring_events_ = mon_update->events()->size();
  tick_stats.group_routes_ = pgwr_event_types.size();
  tick_stats.combined_groups_ = combined_groups.size();
  tick_stats.major_delay_group_routes_ = delayed_group_routes;

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
    ctx::await_all(futures);
    mod.routing_cache_.sync();
    MOTIS_STOP_TIMING(find_alternatives);
    tick_stats.t_find_alternatives_ = MOTIS_TIMING_MS(find_alternatives);
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
    MOTIS_STOP_TIMING(add_alternatives);
    tick_stats.t_add_alternatives_ = MOTIS_TIMING_MS(add_alternatives);
  }

  LOG(info) << "alternatives: " << routing_requests << " routing requests => "
            << alternatives_found << " alternatives";

  tick_stats.routing_requests_ = routing_requests;
  tick_stats.alternatives_found_ = alternatives_found;

  auto removed_group_route_count = 0ULL;
  if (delayed_group_routes > 0) {
    std::vector<passenger_group_with_route_and_localization>
        group_routes_to_remove;
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        if (!cpg.has_major_delay_groups_) {
          continue;
        }

        // remove groups without better alternatives from cpg
        // so that they are not included in the simulation
        // (they remain unchanged)
        utl::erase_if(
            cpg.group_routes_,
            [&](passenger_group_with_route_and_probability const& pgwrap) {
              if (pgwr_event_types.at(pgwrap.pgwr_) !=
                  monitoring_event_type::MAJOR_DELAY_EXPECTED) {
                return false;
              }
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
          if (pgwr_event_types.at(pgwrap.pgwr_) ==
              monitoring_event_type::MAJOR_DELAY_EXPECTED) {
            group_routes_to_remove.emplace_back(
                passenger_group_with_route_and_localization{
                    pgwrap.pgwr_, &cpg.localization_});
            ++tick_stats.major_delay_group_routes_with_alternatives_;
            ++removed_group_route_count;
          }
        }

        if (group_routes_to_remove.size() >= REMOVE_GROUPS_BATCH_SIZE) {
          send_remove_group_routes(sched, uv, group_routes_to_remove,
                                   broken_transfer_infos, tick_stats,
                                   reroute_reason_t::MAJOR_DELAY_EXPECTED);
        }
      }
    }
    send_remove_group_routes(sched, uv, group_routes_to_remove,
                             broken_transfer_infos, tick_stats,
                             reroute_reason_t::MAJOR_DELAY_EXPECTED);
    LOG(info) << "delayed group routes: " << delayed_group_routes
              << ", removed group routes: " << removed_group_route_count
              << " (tick total: " << tick_stats.removed_group_routes_ << ")";
  }

  MOTIS_START_TIMING(passenger_behavior);
  manual_timer sim_timer{"passenger behavior simulation"};
  auto pb = behavior::default_behavior{mod.deterministic_mode_};
  auto const sim_result = simulate_behavior(sched, uv, combined_groups, pb.pb_,
                                            mod.probability_threshold_);
  sim_timer.stop_and_print();
  MOTIS_STOP_TIMING(passenger_behavior);
  tick_stats.t_passenger_behavior_ = MOTIS_TIMING_MS(passenger_behavior);

  LOG(info) << "forecast: " << sim_result.additional_groups_.size()
            << " edges affected";
  LOG(info) << fmt::format(
      "simulation average statistics: alternatives found: {:.2f}, alternatives "
      "picked: {:.2f}, P(best): {:.2f}%, P(2nd best): {:.2f}% ({} group "
      "routes, {} "
      "combined)",
      sim_result.stats_.found_alt_count_avg_,
      sim_result.stats_.picked_alt_count_avg_,
      sim_result.stats_.best_alt_prob_avg_ * 100,
      sim_result.stats_.second_alt_prob_avg_ * 100,
      sim_result.stats_.group_route_count_,
      sim_result.stats_.combined_group_count_);

  if (mod.behavior_stats_file_.is_open() && uv.id_ == 0) {
    fmt::print(mod.behavior_stats_file_,
               "{},{},{},{:.4f},{:.4f},{:.2f},{:.2f}\n",
               static_cast<std::uint64_t>(sched.system_time_),
               sim_result.stats_.group_route_count_,
               sim_result.stats_.combined_group_count_,
               sim_result.stats_.found_alt_count_avg_,
               sim_result.stats_.picked_alt_count_avg_,
               sim_result.stats_.best_alt_prob_avg_ * 100,
               sim_result.stats_.second_alt_prob_avg_ * 100);
  }

  if (mod.calc_load_forecast_) {
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
    if (mod.forecast_file_.is_open() && uv.id_ == 0) {
      scoped_timer const load_forecast_msg_timer{"load forecast to json"};
      mod.forecast_file_ << forecast_msg->to_json(json_format::SINGLE_LINE)
                         << std::endl;
    }
    MOTIS_STOP_TIMING(write_load_forecast);
    tick_stats.t_write_load_forecast_ = MOTIS_TIMING_MS(write_load_forecast);

    MOTIS_START_TIMING(publish_load_forecast);
    if (mod.publish_load_forecast_) {
      ctx::await_all(motis_publish(forecast_msg));
    }
    MOTIS_STOP_TIMING(publish_load_forecast);
    tick_stats.t_publish_load_forecast_ =
        MOTIS_TIMING_MS(publish_load_forecast);

    MOTIS_STOP_TIMING(total_load_forecast);
    tick_stats.t_total_load_forecast_ = MOTIS_TIMING_MS(total_load_forecast);
  }

  MOTIS_START_TIMING(update_tracked_groups);
  scoped_timer const update_tracked_groups_timer{"update tracked groups"};
  update_tracked_groups(sched, uv, sim_result, pgwr_event_types,
                        broken_transfer_infos, pgwr_localizations, tick_stats,
                        reroute_reason_t::REVERT_FORECAST);
  MOTIS_STOP_TIMING(update_tracked_groups);
  tick_stats.t_update_tracked_groups_ = MOTIS_TIMING_MS(update_tracked_groups);

  if (!unbroken_transfers.empty() && mod.revert_forecasts_) {
    revert_forecasts(uv, sched, sim_result, unbroken_transfers,
                     pgwr_localizations);
  }

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
