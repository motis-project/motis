#include "motis/paxforecast/api/apply_measures.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"

#include "motis/core/access/trip_access.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/update_capacity.h"

#include "motis/paxmon/loader/capacities/load_capacities.h"

#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/error.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/paxforecast.h"
#include "motis/paxforecast/simulate_behavior.h"
#include "motis/paxforecast/universe_data.h"
#include "motis/paxforecast/update_tracked_groups.h"

#include "motis/paxforecast/measures/affected_groups.h"
#include "motis/paxforecast/measures/measures.h"

#include "motis/paxforecast/behavior/default_behavior.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::paxmon;
using namespace motis::paxforecast;

namespace motis::paxforecast::api {

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

    // combine groups by (localization, remaining planned journey)
    auto combined =
        mcd::hash_map<mcd::pair<passenger_localization, compact_journey>,
                      combined_passenger_group>{};
    mcd::hash_map<passenger_group_with_route, passenger_localization const*>
        pgwr_localizations;
    for (auto const& [pgwr, loc] : affected_groups.localization_) {
      auto const& pg = uv.passenger_groups_.group(pgwr.pg_);
      auto const& gr = uv.passenger_groups_.route(pgwr);
      auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);
      auto const remaining_planned_journey = get_suffix(sched, cj, loc);
      pgwr_localizations[pgwr] = &loc;
      if (remaining_planned_journey.legs_.empty()) {
        continue;
      }
      auto& cpg = combined[{loc, remaining_planned_journey}];
      cpg.group_routes_.emplace_back(passenger_group_with_route_and_probability{
          pgwr, gr.probability_, pg.passengers_});
      cpg.passengers_ += pg.passengers_;
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
      for (auto const& pgwrap : cpg.group_routes_) {
        for (auto const* mv : affected_groups.measures_.at(pgwrap.pgwr_)) {
          measures_set.insert(mv);
        }
      }
      auto group_measures = mcd::to_vec(measures_set);
      futures.emplace_back(spawn_job_void([&, group_measures] {
        cpg.alternatives_ = find_alternatives(
            uv, sched, mod.routing_cache_, group_measures,
            remaining_planned_journey.destination_station_id(), loc,
            &remaining_planned_journey, false, 61, mod.allow_start_metas_,
            mod.allow_dest_metas_);
      }));
    }
    ctx::await_all(futures);
    mod.routing_cache_.sync();
    alternatives_timer.stop_and_print();
    t_find_alternatives += alternatives_timer.duration_ms();
    total_alternative_routings += combined.size();

    {
      manual_timer alt_trips_timer{"add alternatives to graph"};
      for (auto& [grp_key, cpg] : combined) {
        total_alternatives_found += cpg.alternatives_.size();
        for (auto const& alt : cpg.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, uv, leg.trip_idx_);
          }
        }
      }
      alt_trips_timer.stop_and_print();
      t_add_alternatives_to_graph += alt_trips_timer.duration_ms();
    }

    manual_timer sim_timer{"passenger behavior simulation"};
    auto pb = behavior::default_behavior{mod.deterministic_mode_};
    auto const sim_result = simulate_behavior(sched, uv, combined, pb.pb_,
                                              mod.probability_threshold_);
    sim_timer.stop_and_print();
    t_behavior_simulation += sim_timer.duration_ms();

    manual_timer update_groups_timer{"update groups"};
    tick_statistics tick_stats;
    update_tracked_groups(sched, uv, sim_result, {}, {}, pgwr_localizations,
                          tick_stats, reroute_reason_t::SIMULATION);
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
