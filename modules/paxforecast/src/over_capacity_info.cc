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
                                      simulation_result const& sim_result) {
  over_capacity_info oci;

  for (auto const& [e, additional_groups] : sim_result.additional_groups_) {
    if (!e->is_trip() || !e->has_capacity()) {
      continue;
    }
    auto const capacity = e->capacity();
    auto pdf = get_load_pdf(e->get_pax_connection_info());
    for (auto const& [grp, grp_probability] : additional_groups) {
      add_additional_group(pdf, grp->passengers_, grp_probability);
    }
    if (load_factor_possibly_ge(pdf, capacity, 1.0F)) {
      oci.over_capacity_edges_[e] = edge_over_capacity_info{get_cdf(pdf)};
      for (auto const& trp : e->get_trips(sched)) {
        oci.over_capacity_trips_[trp].emplace_back(e);
      }
    }
  }

  return oci;
}

}  // namespace motis::paxforecast
