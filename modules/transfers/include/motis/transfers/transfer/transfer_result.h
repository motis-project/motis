#pragma once

#include <ostream>
#include <vector>

#include "motis/transfers/transfer/transfer_request.h"
#include "motis/transfers/types.h"

#include "motis/ppr/profile_info.h"

#include "ppr/common/routing_graph.h"

#include "nigiri/types.h"

namespace motis::transfers {

struct transfer_info {
  CISTA_COMPARABLE();
  ::nigiri::duration_t duration_{};
  double distance_{};
};
using transfer_infos = std::vector<transfer_info>;

struct transfer_result {
  CISTA_COMPARABLE();
  nlocation_key_t from_nloc_key_;
  vector<nlocation_key_t> to_nloc_keys_;
  profile_key_t profile_;

  vector<transfer_info> infos_;
};
using transfer_results = std::vector<transfer_result>;

// Routes a single `transfer_request` and returns the corresponding
// `transfer_result`. PPR is used for routing.
transfer_result route_single_request(
    transfer_request const&, ::ppr::routing_graph const&,
    hash_map<profile_key_t, ppr::profile_info> const&);

// Routes a batch of `transfer_request`s and returns the corresponding list of
// `transfer_result`s.
// Equivalent: Calls `route_single_request` for every
// `transfer_request`.
transfer_results route_multiple_requests(
    transfer_requests const&, ::ppr::routing_graph const&,
    hash_map<profile_key_t, ppr::profile_info> const&);

// Returns a new merged `transfer_result` struct.
// Default values used from `lhs` struct.
// Adds `to_nloc_keys_` and corresponding `info` if the `to_nloc_key` is not yet
// considered over `lhs`.
// Limitations:
// - Already existing `transfer_result::info` in `lhs` will not be updated if
// there is a newer one in `rhs`.
// Merge Prerequisites:
// - `lhs.from_nloc_key_ == rhs.from_nloc_key_`
// - `lhs.profile_ == rhs.profile_`
// - `lhs.to_nloc_keys_.size() == lhs.infos_.size()`
// - `rhs.to_nloc_keys_.size() == rhs.infos_.size()`
transfer_result merge(transfer_result const&, transfer_result const&);

// Returns a unique string representation of the given `transfer_result` struct.
string to_key(transfer_result const&);

std::ostream& operator<<(std::ostream&, transfer_info const&);
std::ostream& operator<<(std::ostream&, transfer_result const&);

}  // namespace motis::transfers
