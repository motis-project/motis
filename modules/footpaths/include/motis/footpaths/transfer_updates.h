#pragma once

#include <vector>

#include "boost/thread/mutex.hpp"

#include "nigiri/timetable.h"

#include "motis/footpaths/transfer_requests.h"
#include "ppr/common/routing_graph.h"

using namespace ppr;

namespace motis::footpaths {

void compute_and_update_nigiri_transfers(routing_graph const& rg,
                                         nigiri::timetable tt,
                                         transfer_requests req,
                                         boost::mutex& mutex);

void precompute_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable tt,
    std::vector<transfer_requests> const& transfer_reqs);

}  // namespace motis::footpaths
