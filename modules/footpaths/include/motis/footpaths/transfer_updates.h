#pragma once

#include <vector>

#include "boost/thread/mutex.hpp"

#include "nigiri/timetable.h"

#include "motis/footpaths/transfer_requests.h"

#include "ppr/common/routing_graph.h"
#include "ppr/routing/search.h"

using namespace ppr;
using namespace ppr::routing;

namespace motis::footpaths {

inline location to_location(geo::latlng const& pos) {
  return make_location(pos.lng_, pos.lat_);
}

/**
 * Returns the equivalent OSM_TYPE of PPR.
 * Default: NODE
 */
osm_namespace to_ppr_osm_type(nigiri::osm_type const& t);

/**
 * Creates an input-location struct from a platform-info struct.
 *
 * @param pi the platform-info from which to create an input location.
 * @return an input location struct
 */
input_location pi_to_il(platform_info const& pi);

void compute_and_update_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& req, boost::mutex& mutex);

void precompute_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    std::vector<transfer_requests> const& transfer_reqs);

}  // namespace motis::footpaths
