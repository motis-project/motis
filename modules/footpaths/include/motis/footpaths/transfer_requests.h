#pragma once

#include <map>
#include <string>

#include "motis/footpaths/database.h"
#include "motis/footpaths/state.h"
#include "motis/footpaths/transfers.h"

#include "motis/ppr/profiles.h"

namespace motis::footpaths {

transfer_requests to_transfer_requests(transfer_requests_keys const&,
                                       database&);

transfer_requests_keys generate_transfer_requests_keys(
    state const&, state const&,
    std::map<std::string, ppr::profile_info> const&);

}  // namespace motis::footpaths
