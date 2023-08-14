#pragma once

#include <map>
#include <string>

#include "motis/footpaths/transfers.h"

#include "motis/ppr/profile_info.h"

#include "ppr/common/routing_graph.h"

namespace motis::footpaths {

transfer_results route_multiple_requests(
    transfer_requests const&, ::ppr::routing_graph const&,
    std::map<std::string, ppr::profile_info> const&);

}  // namespace motis::footpaths
