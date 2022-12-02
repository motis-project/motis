#include "motis/paxmon/load_info.h"

#include "utl/pipes.h"

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

inline bool get_possibly_over_capacity(edge const* e, pax_cdf const& cdf) {
  return e->has_capacity() && load_factor_possibly_ge(cdf, e->capacity(), 1.F);
}

inline float get_probability_over_capacity(edge const* e, pax_cdf const& cdf) {
  return e->has_capacity() ? get_pax_gt_probability(cdf, e->capacity()) : 0.F;
}

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf&& pdf, pax_cdf&& cdf, bool updated) {
  auto const possibly_over_capacity = get_possibly_over_capacity(e, cdf);
  auto const probability_over_capacity = get_probability_over_capacity(e, cdf);
  auto const expected_pax = get_expected_load(uv, e->pci_);
  return edge_load_info{
      e,           std::move(pdf),         std::move(cdf),
      updated,     possibly_over_capacity, probability_over_capacity,
      expected_pax};
}

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf const& pdf, pax_cdf const& cdf,
                                   bool updated) {
  auto const possibly_over_capacity = get_possibly_over_capacity(e, cdf);
  auto const probability_over_capacity = get_probability_over_capacity(e, cdf);
  auto const expected_pax = get_expected_load(uv, e->pci_);
  return edge_load_info{e,
                        pdf,
                        cdf,
                        updated,
                        possibly_over_capacity,
                        probability_over_capacity,
                        expected_pax};
}

trip_load_info calc_trip_load_info(universe const& uv, trip const* trp) {
  return trip_load_info{
      trp,
      utl::all(uv.trip_data_.edges(trp))  //
          | utl::transform([&](auto const e) { return e.get(uv); })  //
          | utl::remove_if([](auto const* e) { return !e->is_trip(); })  //
          | utl::transform([&](auto const* e) {
              auto pdf =
                  get_load_pdf(uv.passenger_groups_,
                               uv.pax_connection_info_.group_routes(e->pci_));
              auto cdf = get_cdf(pdf);
              return make_edge_load_info(uv, e, std::move(pdf), std::move(cdf),
                                         false);
            })  //
          | utl::vec()};
}

}  // namespace motis::paxmon
