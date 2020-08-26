#pragma once

#include <iostream>

#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/core/schedule/schedule.h"

namespace motis::rt {

inline void validate_constant_graph(schedule const& sched) {
  motis::logging::manual_timer lb_update("calculating full lower bound graphs");
  auto const full_transfers_lower_bounds_fwd = build_interchange_graph(
      sched.station_nodes_, sched.non_station_node_offset_, sched.route_count_,
      search_dir::FWD);
  auto const full_transfers_lower_bounds_bwd = build_interchange_graph(
      sched.station_nodes_, sched.non_station_node_offset_, sched.route_count_,
      search_dir::BWD);
  auto const full_travel_time_lower_bounds_fwd =
      build_station_graph(sched.station_nodes_, search_dir::FWD);
  auto const full_travel_time_lower_bounds_bwd =
      build_station_graph(sched.station_nodes_, search_dir::BWD);
  lb_update.stop_and_print();

  auto const check_graph = [](constant_graph const& updated_cg,
                              constant_graph const& ref_cg,
                              char const* graph_name) {
    utl::verify(updated_cg.size() >= ref_cg.size(),
                "invalid constant graph size");
    for (auto from = 0ULL; from < ref_cg.size(); ++from) {
      auto const& updated = updated_cg[from];
      for (auto const& ref_se : ref_cg[from]) {
        auto const to = ref_se.to_;
        auto const updated_ce =
            std::find_if(begin(updated), end(updated),
                         [to](auto const& se) { return se.to_ == to; });
        utl::verify(updated_ce != end(updated), "constant graph: missing edge");
        utl::verify(updated_ce->cost_ <= ref_se.cost_,
                    "constant graph: wrong costs: graph={}, from={}, to={}, "
                    "updated={}, ref={}",
                    graph_name, from, to, updated_ce->cost_, ref_se.cost_);
      }
    }
  };

  check_graph(sched.transfers_lower_bounds_fwd_,
              full_transfers_lower_bounds_fwd, "transfers_fwd");
  check_graph(sched.transfers_lower_bounds_bwd_,
              full_transfers_lower_bounds_bwd, "transfers_bwd");
  check_graph(sched.travel_time_lower_bounds_fwd_,
              full_travel_time_lower_bounds_fwd, "time_fwd");
  check_graph(sched.travel_time_lower_bounds_bwd_,
              full_travel_time_lower_bounds_bwd, "time_bwd");
}

}  // namespace motis::rt
