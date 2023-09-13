#pragma once

#include "motis/transfers/transfer/transfer_result.h"
#include "motis/transfers/types.h"

#include "nigiri/footpath.h"
#include "nigiri/types.h"

#include "geo/latlng.h"

namespace motis::transfers {

struct to_nigiri_data {
  // --- nigiri timetable data ---

  ::nigiri::vector_map<::nigiri::location_idx_t, geo::latlng> const& coords_;
  ::nigiri::hash_map<::nigiri::string, ::nigiri::profile_idx_t> const&
      profile_name_to_tt_idx_;

  // --- storage data ---

  hash_map<profile_key_t, string> const& profile_key_to_profile_name_;

  // --- transfer results ---

  transfer_results transfer_results_;
};

struct nigiri_transfers {
  array<mutable_fws_multimap<::nigiri::location_idx_t, ::nigiri::footpath>,
        ::nigiri::kMaxProfiles>
      out_{}, in_{};
};

// Outputs an `nigiri_transfers` struct containing incoming and outgoing
// transfers to/from an `nigiri::location`. By precalculating the
// `nigiri_transfers` struct, importing new transfers into the
// `nigiri::timetable` is efficient.
nigiri_transfers build_nigiri_transfers(to_nigiri_data const&);

}  // namespace motis::transfers
