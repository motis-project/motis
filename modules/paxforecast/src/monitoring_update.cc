#include "motis/paxforecast/monitoring_update.h"

#include <cstdint>
#include <algorithm>
#include <optional>
#include <vector>

#include "fmt/format.h"

#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/fbs_compact_journey_util.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/localization_conv.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"

#include "motis/paxforecast/affected_route_info.h"
#include "motis/paxforecast/paxforecast.h"
#include "motis/paxforecast/revert_forecast.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/universe_data.h"

#include "motis/paxforecast/behavior/default_behavior.h"

using namespace motis::paxmon;
using namespace motis::module;
using namespace motis::logging;

namespace motis::paxforecast {

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

void find_alternatives_set(paxforecast& mod, universe& uv,
                           schedule const& sched, tick_statistics& tick_stats,
                           alternatives_set& alts_set) {
  {
    MOTIS_START_TIMING(find_alternatives);
    scoped_timer const alt_timer{"on_monitoring_event: find alternatives"};
    LOG(info) << "find alternatives: " << alts_set.requests_.size()
              << " routing requests (using cache="
              << mod.routing_cache_.is_open() << ")...";
    tick_stats.routing_requests_ += alts_set.requests_.size();
    alts_set.find(uv, sched, mod.routing_cache_,
                  alternative_routing_options{
                      .use_cache_ = true,
                      .pretrip_interval_length_ = 0,
                      .allow_start_metas_ = mod.allow_start_metas_,
                      .allow_dest_metas_ = mod.allow_dest_metas_});
    MOTIS_STOP_TIMING(find_alternatives);
    tick_stats.t_find_alternatives_ += MOTIS_TIMING_MS(find_alternatives);
  }

  auto alternatives_found = 0ULL;
  {
    MOTIS_START_TIMING(add_alternatives);
    scoped_timer const alt_trips_timer{"add alternatives to graph"};
    for (auto const& req : alts_set.requests_) {
      alternatives_found += req.alternatives_.size();
      for (auto const& alt : req.alternatives_) {
        for (auto const& leg : alt.compact_journey_.legs_) {
          get_or_add_trip(sched, uv, leg.trip_idx_);
        }
      }
    }
    tick_stats.alternatives_found_ += alternatives_found;
    MOTIS_STOP_TIMING(add_alternatives);
    tick_stats.t_add_alternatives_ += MOTIS_TIMING_MS(add_alternatives);
  }

  LOG(info) << "alternatives: " << alts_set.requests_.size()
            << " routing requests => " << alternatives_found << " alternatives";
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

void run_simulation(paxforecast& mod, tick_statistics& tick_stats,
                    alternatives_set& alts_set) {
  MOTIS_START_TIMING(passenger_behavior);
  auto pb = behavior::default_behavior{mod.deterministic_mode_};
  simulate_behavior_for_alternatives(pb.pb_, alts_set);
  MOTIS_STOP_TIMING(passenger_behavior);
  tick_stats.t_passenger_behavior_ += MOTIS_TIMING_MS(passenger_behavior);
}

void update_groups(universe& uv, schedule const& sched,
                   std::vector<affected_route_info> const& affected_routes,
                   alternatives_set const& alts_set,
                   tick_statistics& tick_stats,
                   simulation_options const& options,
                   reroute_reason_t const reason) {
  auto const constexpr REROUTE_BATCH_SIZE = 5'000;

  MOTIS_START_TIMING(update_tracked_groups);

  auto ug_ctx = update_groups_context{.mc_ = message_creator()};
  auto const empty_alts = std::vector<alternative>{};

  auto const send_reroutes = [&]() {
    if (ug_ctx.reroutes_.empty()) {
      return;
    }
    tick_stats.rerouted_group_routes_ += ug_ctx.reroutes_.size();
    LOG(motis::logging::info)
        << "update_groups: sending " << ug_ctx.reroutes_.size() << " reroutes";
    ug_ctx.mc_.create_and_finish(
        MsgContent_PaxMonRerouteGroupsRequest,
        CreatePaxMonRerouteGroupsRequest(
            ug_ctx.mc_, uv.id_, ug_ctx.mc_.CreateVector(ug_ctx.reroutes_))
            .Union(),
        "/paxmon/reroute_groups");
    auto const msg = make_msg(ug_ctx.mc_);
    motis_call(msg)->val();
    ug_ctx.reroutes_.clear();
    ug_ctx.mc_.Clear();
  };

  for (auto const& ar : affected_routes) {
    simulate_behavior_for_route(
        sched, uv, ug_ctx, options, ar,
        alts_set.requests_.at(ar.alts_now_).alternatives_,
        ar.loc_broken_.valid()
            ? alts_set.requests_.at(ar.alts_broken_).alternatives_
            : empty_alts,
        reason);
    if (ug_ctx.reroutes_.size() >= REROUTE_BATCH_SIZE) {
      send_reroutes();
    }
  }

  send_reroutes();

  MOTIS_STOP_TIMING(update_tracked_groups);
  tick_stats.t_update_tracked_groups_ += MOTIS_TIMING_MS(update_tracked_groups);
}

void handle_broken_transfers(paxforecast& mod, universe& uv,
                             schedule const& sched, tick_statistics& tick_stats,
                             PaxMonUpdate const* mon_update) {
  auto const use_uninformed_pax = mod.uninformed_pax_ > 0.F;
  auto affected_routes = std::vector<affected_route_info>{};
  auto alts_set = alternatives_set{};

  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_BROKEN_TRANSFER) {
      continue;
    }

    auto const pgwrap = event_to_pgwrap(event);
    auto loc_now =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id = get_destination_station_id(
        sched, event->group_route()->route()->journey());

    if (pgwrap.probability_ == 0.F) {
      continue;
    }

    auto& ar = affected_routes.emplace_back(affected_route_info{
        .pgwrap_ = pgwrap,
        .destination_station_id_ = destination_station_id,
        .loc_now_ = std::move(loc_now),
        .broken_transfer_info_ =
            from_fbs(sched, event->reachability()->broken_transfer()),
    });

    ar.alts_now_ =
        alts_set.add_request(ar.loc_now_, ar.destination_station_id_);

    if (use_uninformed_pax && ar.broken_transfer_info_) {
      ar.loc_broken_ = localize_broken_transfer(
          sched,
          uv.passenger_groups_.journey(
              uv.passenger_groups_.route(ar.pgwrap_.pgwr_)
                  .compact_journey_index_),
          *ar.broken_transfer_info_);
      ar.alts_broken_ =
          alts_set.add_request(ar.loc_broken_, ar.destination_station_id_);
    }
  }

  if (alts_set.requests_.empty()) {
    return;
  }

  tick_stats.group_routes_ += affected_routes.size();
  tick_stats.combined_groups_ += alts_set.requests_.size();

  find_alternatives_set(mod, uv, sched, tick_stats, alts_set);
  run_simulation(mod, tick_stats, alts_set);
  update_groups(
      uv, sched, affected_routes, alts_set, tick_stats,
      simulation_options{.probability_threshold_ = mod.probability_threshold_,
                         .uninformed_pax_ = mod.uninformed_pax_},
      reroute_reason_t::BROKEN_TRANSFER);
}

void handle_major_delays(paxforecast& mod, universe& uv, schedule const& sched,
                         tick_statistics& tick_stats,
                         PaxMonUpdate const* mon_update) {
  auto const switch_prob = mod.major_delay_switch_;
  auto const stay_prob = 1.F - switch_prob;
  if (switch_prob == 0.F) {
    return;
  }

  auto affected_routes = std::vector<affected_route_info>{};
  auto alts_set = alternatives_set{};

  auto last_pgi = std::numeric_limits<passenger_group_index>::max();
  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_MAJOR_DELAY_EXPECTED) {
      continue;
    }

    auto pgwrap = event_to_pgwrap(event);
    // probability may have changed because of broken transfers in other
    // routes of the same group
    pgwrap.probability_ = uv.passenger_groups_.route(pgwrap.pgwr_).probability_;

    if (pgwrap.probability_ == 0.F) {
      continue;
    }

    if (last_pgi == pgwrap.pgwr_.pg_) {
      // TODO(pablo): major delays for more than one route per group
      //  are currently not supported because processing them separately
      //  breaks things
      LOG(info) << "skipping major delay for group " << pgwrap.pgwr_.pg_
                << " because of multiple delayed routes";
      continue;
    }
    last_pgi = pgwrap.pgwr_.pg_;

    auto& ar = affected_routes.emplace_back(affected_route_info{
        .pgwrap_ = pgwrap,
        .destination_station_id_ = get_destination_station_id(
            sched, event->group_route()->route()->journey()),
        .loc_now_ = from_fbs(sched, event->localization_type(),
                             event->localization())});

    ar.alts_now_ =
        alts_set.add_request(ar.loc_now_, ar.destination_station_id_);
  }

  if (alts_set.requests_.empty()) {
    return;
  }

  find_alternatives_set(mod, uv, sched, tick_stats, alts_set);
  run_simulation(mod, tick_stats, alts_set);
  update_groups(
      uv, sched, affected_routes, alts_set, tick_stats,
      simulation_options{.probability_threshold_ = mod.probability_threshold_,
                         .uninformed_pax_ = stay_prob},
      reroute_reason_t::MAJOR_DELAY_EXPECTED);
}

void handle_unbroken_transfers(paxforecast& mod, universe& uv,
                               schedule const& sched,
                               PaxMonUpdate const* mon_update) {
  auto unbroken_transfers = std::vector<passenger_group_with_route>{};

  for (auto const& event : *mon_update->events()) {
    if (event->type() != PaxMonEventType_REACTIVATED) {
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
  handle_unbroken_transfers(mod, uv, sched, mon_update);

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
