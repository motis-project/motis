#pragma once

#include <map>

#include "motis/ppr/profile_info.h"

#include "motis/parking/parkings.h"
#include "motis/parking/prepare/stations.h"

namespace motis::parking::prepare {

void compute_foot_edges(
    stations const& st, parkings const& park,
    std::string const& footedges_db_file, std::string const& ppr_graph,
    std::size_t edge_rtree_max_size, std::size_t area_rtree_max_size,
    bool lock_rtrees,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    int threads, std::string const& stations_per_parking_file);

}  // namespace motis::parking::prepare
