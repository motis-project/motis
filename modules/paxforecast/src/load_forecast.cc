#include "motis/paxforecast/load_forecast.h"

#include <mutex>

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_parallel_for.h"

#include "motis/paxmon/load_info.h"

using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxforecast {

load_forecast calc_load_forecast(schedule const& sched, universe const& uv,
                                 simulation_result const& sim_result) {
  mcd::hash_map<motis::paxmon::edge const*, edge_load_info> edges;
  mcd::hash_set<trip const*> trips;
  std::mutex mutex;

  LOG(info) << "calc_load_forecast: " << sim_result.additional_groups_.size()
            << " edges with additional groups";

  motis_parallel_for(sim_result.additional_groups_, [&](auto const& entry) {
    auto const e = entry.first;
    if (!e->is_trip()) {
      return;
    }
    if (e->clasz_ != service_class::ICE && e->clasz_ != service_class::IC &&
        e->clasz_ != service_class::OTHER) {
      return;
    }
    auto const& additional_groups = entry.second;
    auto pdf = get_load_pdf(uv.passenger_groups_,
                            uv.pax_connection_info_.group_routes(e->pci_));
    add_additional_groups(pdf, additional_groups);
    auto cdf = get_cdf(pdf);

    std::lock_guard guard{mutex};
    edges.emplace(
        e, make_edge_load_info(uv, e, std::move(pdf), std::move(cdf), true));
    for (auto const& trp : e->get_trips(sched)) {
      trips.emplace(trp);
    }
  });

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
