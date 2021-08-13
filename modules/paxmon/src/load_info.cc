#include "motis/paxmon/load_info.h"

#include "utl/pipes.h"

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

trip_load_info calc_trip_load_info(paxmon_data const& data, trip const* trp) {
  return trip_load_info{
      trp,
      utl::all(data.graph_.trip_data_.at(trp)->edges_)  //
          | utl::remove_if([](auto const e) { return !e->is_trip(); })  //
          | utl::transform([&](auto const e) {
              auto const cdf = get_load_cdf(e->get_pax_connection_info());
              auto const possibly_over_capacity =
                  e->has_capacity() &&
                  load_factor_possibly_ge(cdf, e->capacity(), 1.0F);
              auto const expected_pax =
                  get_expected_load(e->get_pax_connection_info());
              return edge_load_info{e, cdf, false, possibly_over_capacity,
                                    expected_pax};
            })  //
          | utl::vec()};
}

}  // namespace motis::paxmon
