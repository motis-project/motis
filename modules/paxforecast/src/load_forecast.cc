#include "motis/paxforecast/load_forecast.h"

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/paxmon/load_info.h"

using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxforecast {

load_forecast calc_load_forecast(schedule const& sched, universe const& uv) {
  mcd::hash_map<motis::paxmon::edge const*, edge_load_info> edges;
  mcd::hash_set<trip const*> trips;

  load_forecast lfc;
  lfc.trips_ = utl::to_vec(trips, [&](auto const trp) {
    return trip_load_info{
        trp,
        utl::all(uv.trip_data_.edges(trp))  //
            | utl::transform([&](auto const& e) { return e.get(uv); })  //
            | utl::remove_if([](auto const e) { return !e->is_trip(); })  //
            | utl::transform([&](auto const e) {
                auto const it = edges.find(e);
                if (it != end(edges)) {
                  return it->second;
                } else {
                  auto pdf = get_load_pdf(
                      uv.passenger_groups_,
                      uv.pax_connection_info_.group_routes(e->pci_));
                  auto cdf = get_cdf(pdf);
                  return make_edge_load_info(uv, e, std::move(pdf),
                                             std::move(cdf), false);
                }
              })  //
            | utl::vec()};
  });

  return lfc;
}

}  // namespace motis::paxforecast
