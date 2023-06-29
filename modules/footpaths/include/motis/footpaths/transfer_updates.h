#pragma once

#include <vector>

#include "nigiri/timetable.h"

#include "motis/footpaths/transfer_requests.h"
#include "ppr/common/routing_graph.h"

using namespace ppr;

namespace motis::footpaths {

void update_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable tt,
    std::vector<transfer_requests> const& transfer_reqs);

}
