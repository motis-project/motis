#pragma once

#include "motis/footpaths/state.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profiles.h"

namespace motis::footpaths {

transfer_requests to_transfer_requests(
    transfer_requests_keys const&, hash_map<nlocation_key_t, platform> const&);

transfer_requests_keys generate_transfer_requests_keys(
    state const&, state const&,
    hash_map<profile_key_t, ppr::profile_info> const&,
    bool const /* old_to_old */);

}  // namespace motis::footpaths
