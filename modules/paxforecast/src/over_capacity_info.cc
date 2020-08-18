#include "motis/paxforecast/over_capacity_info.h"

using namespace motis::paxmon;

namespace motis::paxforecast {

std::uint16_t additional_passengers(
    std::vector<std::pair<passenger_group const*, float>> const&
        additional_groups,
    float probability) {
  std::uint16_t count = 0;
  for (auto const& [grp, grp_probability] : additional_groups) {
    if (grp_probability >= probability) {
      count += grp->passengers_;
    }
  }
  return count;
}

over_capacity_info calc_over_capacity(schedule const& sched,
                                      simulation_result const& sim_result,
                                      float probability) {
  over_capacity_info oci;
  oci.probability_ = probability;

  for (auto const& [e, additional_groups] : sim_result.additional_groups_) {
    if (!e->is_trip() || !e->has_capacity()) {
      continue;
    }
    auto const capacity = e->capacity();
    auto const current_pax = e->passengers(probability);
    auto const additional_pax =
        additional_passengers(additional_groups, probability);
    auto const total_pax = current_pax + additional_pax;
    if (total_pax > capacity) {
      oci.over_capacity_edges_[e] =
          edge_over_capacity_info{current_pax, additional_pax};
      for (auto const& trp : e->get_trips(sched)) {
        oci.over_capacity_trips_[trp].emplace_back(e);
      }
    }
  }

  return oci;
}

}  // namespace motis::paxforecast
