#include "motis/paxmon/access/groups.h"

#include <optional>

#include "utl/verify.h"

#include "motis/paxmon/access/journeys.h"
#include "motis/paxmon/build_graph.h"

namespace motis::paxmon {

void update_estimated_delay(
    group_route& gr,
    add_group_route_to_graph_result const& add_to_graph_result) {
  if (add_to_graph_result.has_valid_times() &&
      gr.planned_arrival_time_ != INVALID_TIME) {
    gr.estimated_delay_ = static_cast<std::int16_t>(
        static_cast<int>(add_to_graph_result.current_arrival_time_) -
        static_cast<int>(gr.planned_arrival_time_));
  }
}

passenger_group* add_passenger_group(universe& uv, schedule const& sched,
                                     capacity_maps const& caps,
                                     temp_passenger_group const& tpg,
                                     bool const log,
                                     pci_log_reason_t const reason) {
  auto* pg = uv.passenger_groups_.add(
      make_passenger_group(tpg.source_, tpg.passengers_));

  for (auto const& tgr : tpg.routes_) {
    add_group_route(uv, sched, caps, pg->id_, tgr, false, log, reason);
  }

  return pg;
}

add_group_route_result add_group_route(universe& uv, schedule const& sched,
                                       capacity_maps const& caps,
                                       passenger_group_index const pgi,
                                       temp_group_route const& tgr,
                                       bool const override_probabilities,
                                       bool const log,
                                       pci_log_reason_t const reason) {
  auto probability = tgr.probability_;
  auto planned_arrival_time = tgr.planned_arrival_time_;
  auto routes = uv.passenger_groups_.routes(pgi);

  auto const update_route = [&](group_route& gr) {
    auto const previous_probability = gr.probability_;
    auto const new_probability =
        override_probabilities ? probability : gr.probability_ + probability;
    utl::verify(new_probability >= -0.1 && new_probability <= 1.1,
                "paxmon::add_group_route: new probability = {} (previous = "
                "{}, added = {}), existing route = {}",
                new_probability, previous_probability, probability,
                gr.local_group_route_index_);
    gr.probability_ = std::clamp(new_probability, 0.F, 1.F);
    if (gr.disabled_) {
      auto const add_to_graph_result = add_group_route_to_graph(
          sched, caps, uv, uv.passenger_groups_.group(pgi), gr, log, reason);
      gr.disabled_ = false;
      update_estimated_delay(gr, add_to_graph_result);
    }
    return add_group_route_result{{pgi, gr.local_group_route_index_},
                                  false,
                                  previous_probability,
                                  gr.probability_};
  };

  if (tgr.index_.has_value() && tgr.journey_.legs_.empty()) {
    auto const route_idx = tgr.index_.value();
    utl::verify(route_idx < routes.size(),
                "paxmon::add_group_route: invalid route index");
    return update_route(routes.at(route_idx));
  }

  utl::verify(!tgr.journey_.legs().empty(),
              "paxmon::add_group_route: empty journey");

  // check if group route already exists
  for (auto& gr : routes) {
    if (uv.passenger_groups_.journey(gr.compact_journey_index_) ==
        tgr.journey_) {
      return update_route(gr);
    }
  }

  // add new group route
  utl::verify(probability >= -0.1 && probability <= 1.1,
              "paxmon::add_group_route: invalid probability = {}", probability);
  probability = std::clamp(probability, 0.F, 1.F);
  auto const fws_cj = add_compact_journey(uv, tgr.journey_);
  auto const lgr_index = static_cast<local_group_route_index>(routes.size());
  auto const gre_index = static_cast<group_route_edges_index>(
      uv.passenger_groups_.route_edges_.emplace_back().index());
  if (planned_arrival_time == INVALID_TIME) {
    planned_arrival_time = tgr.journey_.scheduled_arrival_time();
  }
  routes.emplace_back(
      make_group_route(fws_cj.index(), lgr_index, gre_index, probability,
                       tgr.planned_, tgr.source_flags_, planned_arrival_time));
  auto const pgwr = passenger_group_with_route{pgi, lgr_index};
  auto& gr = uv.passenger_groups_.route(pgwr);
  auto const add_to_graph_result = add_group_route_to_graph(
      sched, caps, uv, uv.passenger_groups_.group(pgi), gr, log, reason);
  update_estimated_delay(gr, add_to_graph_result);
  return {pgwr, true, 0, probability};
}

void remove_passenger_group(universe& uv, schedule const& sched,
                            passenger_group_index pgi, bool log,
                            pci_log_reason_t const reason) {
  for (auto& route : uv.passenger_groups_.routes(pgi)) {
    remove_group_route(uv, sched, {pgi, route.local_group_route_index_}, log,
                       reason);
  }
  uv.passenger_groups_.release(pgi);
}

void remove_group_route(universe& uv, schedule const& sched,
                        passenger_group_with_route pgwr, bool log,
                        pci_log_reason_t const reason) {
  auto& route = uv.passenger_groups_.route(pgwr);
  remove_group_route_from_graph(uv, sched, uv.passenger_groups_.group(pgwr.pg_),
                                route, log, reason);
  route.probability_ = 0;
  route.disabled_ = true;
}

}  // namespace motis::paxmon
