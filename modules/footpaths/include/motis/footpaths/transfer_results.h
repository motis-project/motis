#pragma once

#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profile_info.h"

#include "ppr/common/routing_graph.h"

namespace motis::footpaths {

transfer_results route_multiple_requests(
    transfer_requests const&, ::ppr::routing_graph const&,
    hash_map<profile_key_t, ppr::profile_info> const&);

}  // namespace motis::footpaths
