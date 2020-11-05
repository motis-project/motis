#include "motis/paxforecast/load_forecast.h"

#include <mutex>

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_parallel_for.h"

using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxforecast {

load_forecast calc_load_forecast(schedule const& sched, paxmon_data const& data,
                                 simulation_result const& sim_result) {
  mcd::hash_map<motis::paxmon::edge const*, edge_forecast> edges;
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
    auto pdf = get_load_pdf(e->get_pax_connection_info());
    add_additional_groups(pdf, additional_groups);
    auto const cdf = get_cdf(pdf);
    auto const possibly_over_capacity =
        e->has_capacity() && load_factor_possibly_ge(cdf, e->capacity(), 1.0F);
    auto const expected_pax = get_expected_load(e->get_pax_connection_info());

    std::lock_guard guard{mutex};
    edges.emplace(
        e, edge_forecast{e, cdf, true, possibly_over_capacity, expected_pax});
    for (auto const& trp : e->get_trips(sched)) {
      trips.emplace(trp);
    }
  });

  load_forecast lfc;
  lfc.trips_ = utl::to_vec(trips, [&](auto const trp) {
    return trip_forecast{
        trp,
        utl::all(data.graph_.trip_data_.at(trp)->edges_)  //
            | utl::remove_if([](auto const e) { return !e->is_trip(); })  //
            | utl::transform([&](auto const e) {
                auto const it = edges.find(e);
                if (it != end(edges)) {
                  return it->second;
                } else {
                  auto const cdf = get_load_cdf(e->get_pax_connection_info());
                  auto const possibly_over_capacity =
                      e->has_capacity() &&
                      load_factor_possibly_ge(cdf, e->capacity(), 1.0F);
                  auto const expected_pax =
                      get_expected_load(e->get_pax_connection_info());
                  return edge_forecast{e, cdf, false, possibly_over_capacity,
                                       expected_pax};
                }
              })  //
            | utl::vec()};
  });

  return lfc;
}

}  // namespace motis::paxforecast
