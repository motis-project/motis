#pragma once

#include <map>

#include "motis/ppr/profile_info.h"

#include "motis/parking/foot_edge_task.h"

#include "motis/parking/parkings.h"
#include "motis/parking/stations.h"

namespace motis::parking {

void compute_foot_edges_direct(
    database& db, std::vector<foot_edge_task> const& tasks,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    std::string const& ppr_graph, std::size_t edge_rtree_max_size,
    std::size_t area_rtree_max_size, bool lock_rtrees, int threads,
    bool ppr_exact);

void compute_foot_edges_via_module(
    database& db, std::vector<foot_edge_task> const& tasks,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    bool ppr_exact);

}  // namespace motis::parking
