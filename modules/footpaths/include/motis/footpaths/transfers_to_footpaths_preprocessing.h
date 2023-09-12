#pragma once

#include <utility>

#include "motis/footpaths/transfer/transfer_result.h"
#include "motis/footpaths/types.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace motis::footpaths {

struct transfer_preprocessing_data {
  // tt data
  ::nigiri::vector_map<::nigiri::location_idx_t, geo::latlng> const& coords_;
  ::nigiri::hash_map<::nigiri::string, ::nigiri::profile_idx_t> const&
      profiles_;

  // db data
  hash_map<profile_key_t, string> const& profile_key_to_profile_name;

  // transfers data
  transfer_results transfer_results_;
};

struct preprocessed_transfers {
  array<mutable_fws_multimap<::nigiri::location_idx_t, ::nigiri::footpath>,
        ::nigiri::kMaxProfiles>
      out_{}, in_{};
};

preprocessed_transfers to_preprocessed_footpaths(
    transfer_preprocessing_data const&);

}  // namespace motis::footpaths
