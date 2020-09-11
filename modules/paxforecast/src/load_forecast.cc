#include "motis/paxforecast/load_forecast.h"

#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

using namespace motis::paxmon;

namespace motis::paxforecast {

load_forecast calc_load_forecast(schedule const& sched, paxmon_data const& data,
                                 simulation_result const& sim_result) {
  mcd::hash_map<motis::paxmon::edge const*, edge_forecast> edges;
  mcd::hash_set<trip const*> trips;

  for (auto const& [e, additional_groups] : sim_result.additional_groups_) {
    if (!e->is_trip()) {
      continue;
    }
    auto pdf = get_load_pdf(e->get_pax_connection_info());
    for (auto const& [grp, grp_probability] : additional_groups) {
      add_additional_group(pdf, grp->passengers_, grp_probability);
    }
    auto const cdf = get_cdf(pdf);
    auto const possibly_over_capacity =
        e->has_capacity() && load_factor_possibly_ge(pdf, e->capacity(), 1.0F);
    edges.emplace(e, edge_forecast{e, cdf, true, possibly_over_capacity});
    for (auto const& trp : e->get_trips(sched)) {
      trips.emplace(trp);
    }
  }

  load_forecast lfc;
  lfc.trips_ = utl::to_vec(trips, [&](auto const trp) {
    return trip_forecast{
        trp,
        utl::to_vec(data.graph_.trip_data_.at(trp)->edges_, [&](auto const e) {
          auto const it = edges.find(e);
          if (it != end(edges)) {
            return it->second;
          } else {
            auto const cdf = get_load_cdf(e->get_pax_connection_info());
            auto const possibly_over_capacity =
                e->has_capacity() &&
                load_factor_possibly_ge(cdf, e->capacity(), 1.0F);
            return edge_forecast{e, cdf, false, possibly_over_capacity};
          }
        })};
  });

  return lfc;
}

}  // namespace motis::paxforecast
