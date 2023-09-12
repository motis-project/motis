#pragma once

#include <ostream>
#include <vector>

#include "motis/footpaths/platform/platform.h"
#include "motis/footpaths/platform/platform_index.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profile_info.h"

namespace motis::footpaths {

struct transfer_request_keys {
  CISTA_COMPARABLE();
  nlocation_key_t from_nloc_key_;
  vector<nlocation_key_t> to_nloc_keys_;
  profile_key_t profile_;
};
using transfer_requests_keys = std::vector<transfer_request_keys>;

struct transfer_request {
  platform transfer_start_;
  nlocation_key_t from_nloc_key_;

  platforms transfer_targets_;
  vector<nlocation_key_t> to_nloc_keys_;

  profile_key_t profile_;
};
using transfer_requests = std::vector<transfer_request>;

struct treq_k_generation_data {
  struct matched_nloc_pf_data {
    platform_index const& matched_pfs_idx_;
    vector<nlocation_key_t> const& nloc_keys_;
    bool set_matched_pfs_idx_;
  } old_, update_;

  hash_map<profile_key_t, ppr::profile_info> profile_key_to_profile_info_;
};

struct transfer_request_options {
  // old_to_old: build transfer requests from already processed (matched
  // platforms) in old_state; use if profiles_hash has been changed
  bool old_to_old_;
};

// Creates a list of `transfer_request` struct from the given list of
// `transfer_request_keys` struct and returns it. Keys are replaced by the
// matched platform.
// Requirement: X must only contain nigiri locations that have
// been successfully matched to an OSM platform.
transfer_requests to_transfer_requests(
    transfer_requests_keys const&, hash_map<nlocation_key_t, platform> const&);

// Generates new `transfer_request_keys` based on matched platforms in the old
// and update state. List of `transfer_request_keys` are always created for the
// following combinations:
// - from old (known matches) to update (new matches)
// - from update (new matches) to old (old matches)
// - from update (new matches) to update (new matches)
// Depending on the given options, the additional creation for the following
// combination is also possible:
// - from old (known matches) to old (known matches)
// The last combination is usually ignored, because the resulting list of
// `transfer_request_keys` are already stored in the storage. An update of these
// `transfer_request_keys` is necessary if the profiles (from PPR) necessary for
// the creation of the `transfer_request_keys` have changed (compared to
// previous applications).
transfer_requests_keys generate_transfer_requests_keys(
    treq_k_generation_data const&, transfer_request_options const&);

// Returns the new merged `transfer_request_keys` struct.
// Default values used from `lhs` struct.
// Adds `to_nloc_keys_` from `rhs` if `nloc_key` is not yet considered over.
// `lhs`.
// Merge Prerequisites:
// - both `transfer_request_keys` structs have the same `from_nloc_key_`
// - both `transfer_request_keys` structs have the same `profile_key_t`
transfer_request_keys merge(transfer_request_keys const& /* lhs */,
                            transfer_request_keys const& /* rhs */);

// Returns a unique string representation of the given `transfer_request_keys`
// struct.
string to_key(transfer_request_keys const&);

// Returns a unique string representation of the given `transfer_request`
// struct.
string to_key(transfer_request const&);

std::ostream& operator<<(std::ostream&, transfer_request const&);
std::ostream& operator<<(std::ostream&, transfer_request_keys const&);

}  // namespace motis::footpaths
