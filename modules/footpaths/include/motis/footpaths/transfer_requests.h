#pragma once

#include <map>
#include <string>

#include "motis/footpaths/state.h"
#include "motis/footpaths/transfers.h"

#include "motis/ppr/profiles.h"

namespace motis::footpaths {

transfer_requests generate_new_all_reachable_pairs_requests(
    state const&, state const&,
    std::map<std::string, ppr::profile_info> const&);

}  // namespace motis::footpaths
