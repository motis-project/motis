#pragma once

#include <vector>

#include "boost/thread/mutex.hpp"

#include "nigiri/timetable.h"

#include "motis/footpaths/transfer_requests.h"

#include "ppr/common/routing_graph.h"
#include "ppr/routing/search.h"

namespace motis::footpaths {

void compute_and_update_nigiri_transfers(
    ::ppr::routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& req, boost::mutex& mutex);

void precompute_nigiri_transfers(
    ::ppr::routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    std::vector<transfer_requests> const& transfer_reqs);

}  // namespace motis::footpaths
