#include "motis/paxforecast/api/apply_measures.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"
#include "motis/core/common/timing.h"

#include "motis/core/access/trip_access.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/update_capacity.h"

#include "motis/paxmon/loader/capacities/load_capacities.h"

#include "motis/paxforecast/affected_route_info.h"
#include "motis/paxforecast/error.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/paxforecast.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/universe_data.h"

#include "motis/paxforecast/measures/affected_groups.h"
#include "motis/paxforecast/measures/measures.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::paxmon;
using namespace motis::paxforecast;

namespace motis::paxforecast::api {

using sim_combined_routes_t = mcd::hash_map<
    mcd::pair<passenger_localization, compact_journey /* remaining journey */>,
    std::vector<std::uint32_t /* index in affected_routes */>>;

void apply_update_capacities_measure(universe& uv, schedule const& sched,
                                     measures::update_capacities const& m) {
  auto& caps = uv.capacity_maps_;

  // reset existing capacity data
  if (m.remove_existing_trip_capacities_) {
    caps.trip_capacity_map_.clear();
  }
  if (m.remove_existing_category_capacities_) {
    caps.category_capacity_map_.clear();
  }
  if (m.remove_existing_vehicle_capacities_) {
    caps.vehicle_capacity_map_.clear();
  }
  if (m.remove_existing_trip_formations_) {
    caps.trip_formation_map_.clear();
    caps.trip_uuid_map_.clear();
    caps.uuid_trip_map_.clear();
  }
  if (m.remove_existing_gattung_capacities_) {
    caps.gattung_capacity_map_.clear();
  }
  if (m.remove_existing_baureihe_capacities_) {
    caps.baureihe_capacity_map_.clear();
  }
  if (m.remove_existing_vehicle_group_capacities_) {
    caps.vehicle_group_capacity_map_.clear();
  }
  if (m.remove_existing_overrides_) {
    caps.override_map_.clear();
  }

  // load new capacity data
  for (auto const& file_content : m.file_contents_) {
    paxmon::loader::capacities::load_capacities(sched, caps, file_content);
  }

  // update all trip capacities
  update_all_trip_capacities(uv, sched, m.track_trip_updates_);
}

void apply_override_capacity_measure(universe& uv, schedule const& sched,
                                     measures::override_capacity const& m) {
  auto& caps = uv.capacity_maps_;
  auto const tid = get_cap_trip_id(m.trip_id_);

  if (m.sections_.empty()) {
    caps.override_map_.erase(tid);
  } else {
    caps.override_map_[tid] = m.sections_;
  }

  if (auto const* trp = find_trip(sched, m.trip_id_); trp != nullptr) {
    update_trip_capacity(uv, sched, trp, true);
  }
}

bool contains_trip(alternative const& alt, extern_trip const& searched_trip) {
  return std::any_of(begin(alt.journey_.trips_), end(alt.journey_.trips_),
                     [&](journey::trip const& jt) {
                       return jt.extern_trip_ == searched_trip;
                     });
}

bool is_recommended(alternative const& alt,
                    measures::trip_recommendation const& m) {
  // TODO(pablo): check interchange stop
  return contains_trip(alt, m.recommended_trip_);
}

void check_measures(
    alternative& alt,
    mcd::vector<measures::measure_variant const*> const& group_measures) {
  for (auto const* mv : group_measures) {
    std::visit(
        utl::overloaded{//
                        [&](measures::trip_recommendation const& m) {
                          if (is_recommended(alt, m)) {
                            alt.is_recommended_ = true;
                          }
                        },
                        [&](measures::trip_load_information const& m) {
                          // TODO(pablo): handle case where load
                          // information for multiple trips in the
                          // journey is available
                          if (contains_trip(alt, m.trip_)) {
                            alt.load_info_ = m.level_;
                          }
                        },
                        [&](measures::trip_load_recommendation const& m) {
                          for (auto const& tll : m.full_trips_) {
                            if (contains_trip(alt, tll.trip_)) {
                              alt.load_info_ = tll.level_;
                            }
                          }
                          for (auto const& tll : m.recommended_trips_) {
                            if (contains_trip(alt, tll.trip_)) {
                              alt.load_info_ = tll.level_;
                              alt.is_recommended_ = true;
                            }
                          }
                        }},
        *mv);
  }
}

template <typename PassengerBehavior>
void sim_and_update_groups(
    paxforecast& mod, universe& uv, schedule const& sched,
    PassengerBehavior& pb,
    measures::affected_groups_info const& affected_groups,
    std::vector<affected_route_info> const& affected_routes,
    alternatives_set const& alts_set, sim_combined_routes_t const& combined,
    tick_statistics& tick_stats) {
  auto const constexpr REROUTE_BATCH_SIZE = 5'000;

  MOTIS_START_TIMING(update_tracked_groups);

  auto const options =
      simulation_options{.probability_threshold_ = mod.probability_threshold_,
                         .uninformed_pax_ = mod.uninformed_pax_};
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

  for (auto const& ce : combined) {
    auto const& remaining_planned_journey = ce.first.second;
    auto const& affected_route_indices = ce.second;
    if (affected_route_indices.empty()) {
      continue;
    }

    auto const& first_affected_route =
        affected_routes.at(affected_route_indices.front());
    auto const& group_measures =
        affected_groups.measures_.at(first_affected_route.pgwrap_.pgwr_);
    auto alts =
        alts_set.requests_.at(first_affected_route.alts_now_).alternatives_;

    for (auto& alt : alts) {
      check_measures(alt, group_measures);
      alt.is_original_ = (alt.compact_journey_ == remaining_planned_journey);
    }

    simulate_behavior_for_alternatives(pb, alts);

    for (auto const ar_idx : affected_route_indices) {
      auto const& ar = affected_routes.at(ar_idx);
      simulate_behavior_for_route(sched, uv, ug_ctx, options, ar, alts,
                                  empty_alts, reroute_reason_t::SIMULATION);
      if (ug_ctx.reroutes_.size() >= REROUTE_BATCH_SIZE) {
        send_reroutes();
      }
    }
  }

  send_reroutes();

  MOTIS_STOP_TIMING(update_tracked_groups);
  tick_stats.t_update_tracked_groups_ += MOTIS_TIMING_MS(update_tracked_groups);
}

msg_ptr apply_measures(paxforecast& mod, paxmon_data& data,
                       msg_ptr const& msg) {
  scoped_timer const all_timer{"apply_measures"};
  auto const req = motis_content(PaxForecastApplyMeasuresRequest, msg);
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  // update measures
  LOG(info) << "parse measures";
  auto const new_ms = from_fbs(sched, req->measures());
  LOG(info) << "get measure storage for universe " << req->universe();
  auto& uv_storage = mod.universe_storage_.get(req->universe());
  auto& measures = uv_storage.measures_;
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
                                    req->include_after_trip_load_info(),
                                    req->include_trips_with_unchanged_load());
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
  auto t_update_capacities = 0.;

  // simulate passenger behavior with measures
  for (auto const& [t, ms] : measures) {
    scoped_timer const measure_timer{"measure"};
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
      using namespace motis::ris;
      manual_timer rt_timer{"applying rt updates"};
      auto rt_lock =
          mod.lock_resources({{uv.schedule_res_id_, ctx::access_t::WRITE}});
      message_creator mc;
      std::vector<flatbuffers::Offset<RISInputMessage>> rims;
      for (auto const& m : ms) {
        if (std::holds_alternative<measures::rt_update>(m)) {
          auto const rtum = std::get<measures::rt_update>(m);
          rims.emplace_back(CreateRISInputMessage(
              mc, rtum.type_, mc.CreateString(rtum.content_)));
        }
      }
      mc.create_and_finish(
          MsgContent_RISApplyRequest,
          CreateRISApplyRequest(mc, uv.schedule_res_id_, mc.CreateVector(rims))
              .Union(),
          "/ris/apply");
      auto const msg = motis_call(make_msg(mc))->val();
      auto const result = motis_content(RISApplyResponse, msg);
      rt_timer.stop_and_print();
      t_rt_updates += rt_timer.duration_ms();
      if (result->failed() != 0) {
        LOG(warn) << "apply_measures: applying rt updates failed: "
                  << result->successful() << " successful, " << result->failed()
                  << " failed";
        throw std::system_error{error::invalid_rt_update_message};
      }
    }

    manual_timer update_capacities_timer{"update capacities"};
    for (auto const& m : ms) {
      if (std::holds_alternative<measures::update_capacities>(m)) {
        auto const ucm = std::get<measures::update_capacities>(m);
        apply_update_capacities_measure(uv, sched, ucm);
      } else if (std::holds_alternative<measures::override_capacity>(m)) {
        auto const ocm = std::get<measures::override_capacity>(m);
        apply_override_capacity_measure(uv, sched, ocm);
      }
    }
    update_capacities_timer.stop_and_print();
    t_update_capacities += update_capacities_timer.duration_ms();

    auto const loc_time = t + req->preparation_time();
    manual_timer get_affected_groups_timer{"get_affected_groups"};
    auto const affected_groups =
        measures::get_affected_groups(sched, uv, loc_time, ms);
    get_affected_groups_timer.stop_and_print();
    t_get_affected_groups += get_affected_groups_timer.duration_ms();

    total_affected_groups += affected_groups.measures_.size();
    LOG(info) << "affected groups: " << affected_groups.measures_.size();

    auto affected_routes = std::vector<affected_route_info>{};
    auto alts_set = alternatives_set{};
    auto combined = sim_combined_routes_t{};

    for (auto const& [pgwr, loc] : affected_groups.localization_) {
      auto const& pg = uv.passenger_groups_.group(pgwr.pg_);
      auto const& gr = uv.passenger_groups_.route(pgwr);
      auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);
      auto const remaining_planned_journey = get_suffix(sched, cj, loc);

      if (remaining_planned_journey.legs_.empty()) {
        continue;
      }

      auto& ar = affected_routes.emplace_back(affected_route_info{
          .pgwrap_ =
              passenger_group_with_route_and_probability{
                  .pgwr_ = pgwr,
                  .probability_ = gr.probability_,
                  .passengers_ = pg.passengers_},
          .destination_station_id_ = cj.destination_station_id(),
          .loc_now_ = loc,
      });

      ar.alts_now_ =
          alts_set.add_request(ar.loc_now_, ar.destination_station_id_);

      combined[{loc, remaining_planned_journey}].emplace_back(
          static_cast<std::uint32_t>(affected_routes.size() - 1));
    }

    manual_timer alternatives_timer{"apply_measures: find alternatives"};
    alts_set.find(uv, sched, mod.routing_cache_,
                  alternative_routing_options{
                      .use_cache_ = false,
                      .pretrip_interval_length_ = 61,
                      .allow_start_metas_ = mod.allow_start_metas_,
                      .allow_dest_metas_ = mod.allow_dest_metas_});
    alternatives_timer.stop_and_print();
    t_find_alternatives += alternatives_timer.duration_ms();
    total_alternative_routings += alts_set.requests_.size();

    {
      manual_timer alt_trips_timer{"add alternatives to graph"};
      for (auto const& req : alts_set.requests_) {
        total_alternatives_found += req.alternatives_.size();
        for (auto const& alt : req.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, uv, leg.trip_idx_);
          }
        }
      }
      alt_trips_timer.stop_and_print();
      t_add_alternatives_to_graph += alt_trips_timer.duration_ms();
    }

    manual_timer update_groups_timer{"sim + update groups"};
    auto tick_stats = tick_statistics{};
    sim_and_update_groups(mod, uv, sched, *mod.behavior_, affected_groups,
                          affected_routes, alts_set, combined, tick_stats);

    update_groups_timer.stop_and_print();
    t_update_groups += update_groups_timer.duration_ms();
    uv_storage.metrics_.add(sched.system_time_, now(), tick_stats);
  }

  manual_timer update_tracker_timer{"update tracker"};
  auto [mc, fb_updates] = uv.update_tracker_.finish_updates();
  update_tracker_timer.stop_and_print();
  t_update_tracker = update_tracker_timer.duration_ms();

  auto const paxmon_tick_stats = uv.update_tracker_.get_tick_statistics();
  auto group_routes_broken = 0ULL;
  auto group_routes_with_major_delay = 0ULL;
  for (auto const& pmts : paxmon_tick_stats) {
    group_routes_broken += pmts.broken_group_routes_;
    group_routes_with_major_delay += pmts.major_delay_group_routes_;
  }

  mc.create_and_finish(
      MsgContent_PaxForecastApplyMeasuresResponse,
      CreatePaxForecastApplyMeasuresResponse(
          mc,
          CreatePaxForecastApplyMeasuresStatistics(
              mc, measure_time_points, total_measures_applied,
              total_affected_groups, total_alternative_routings,
              total_alternatives_found, group_routes_broken,
              group_routes_with_major_delay, t_rt_updates,
              t_get_affected_groups, t_find_alternatives,
              t_add_alternatives_to_graph, t_behavior_simulation,
              t_update_groups, t_update_tracker, t_update_capacities),
          fb_updates)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxforecast::api
