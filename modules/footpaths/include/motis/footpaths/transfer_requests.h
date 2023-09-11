#pragma once

#include "motis/footpaths/platform/platform.h"
#include "motis/footpaths/platform/platform_index.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profile_info.h"

namespace motis::footpaths {

struct treq_k_generation_data {
  struct matched_nloc_pf_data {
    platform_index const& matched_pfs_idx_;
    vector<nlocation_key_t> const& nloc_keys_;
    bool set_matched_pfs_idx_;
  } old_, update_;

  hash_map<profile_key_t, ppr::profile_info> profile_key_to_profile_info_;
};

struct transfer_request_options {
  bool old_to_old_;
};

transfer_requests to_transfer_requests(
    transfer_requests_keys const&, hash_map<nlocation_key_t, platform> const&);

transfer_requests_keys generate_transfer_requests_keys(
    treq_k_generation_data const&, transfer_request_options const&);

}  // namespace motis::footpaths
