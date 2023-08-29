#pragma once

#include <utility>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/footpaths/state.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

struct preprocessing_data {
  // tt data
  nigiri::vector_map<nigiri::location_idx_t, geo::latlng> const& coords_;
  nigiri::hash_map<nigiri::string, nigiri::profile_idx_t> const& profiles_;

  // db data
  hash_map<profile_key_t, string> const& key_to_name_;

  // fps data
  state const& old_state_;
  state const& update_state_;
};

struct preprocessed_footpaths {
  array<mutable_fws_multimap<nigiri::location_idx_t, nigiri::footpath>,
        nigiri::kMaxProfiles>
      out_{}, in_{};
};

preprocessed_footpaths to_preprocessed_footpaths(preprocessing_data const&);

}  // namespace motis::footpaths
