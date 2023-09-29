#include "motis/transfers/transfer_restrictions/helper.h"

namespace n = ::nigiri;

namespace motis::transfers::restrictions {

hash_map<std::string_view, n::location_idx_t>
get_parent_location_name_to_idx_mapping(n::timetable& tt) {
  auto mapping = hash_map<std::string_view, n::location_idx_t>{};
  auto seen_parents = set<n::location_idx_t>{};

  for (auto loc_idx = n::location_idx_t{0U};
       loc_idx < tt.locations_.ids_.size(); ++loc_idx) {
    auto parent_loc_idx = tt.locations_.parents_.at(loc_idx);

    // only consider locations with a valid parent location idx
    if (parent_loc_idx == n::location_idx_t::invalid()) {
      continue;
    }

    // process a parent location at most once
    if (seen_parents.count(parent_loc_idx) == 1) {
      continue;
    }

    mapping.emplace(tt.locations_.names_.at(parent_loc_idx).view(),
                    parent_loc_idx);
    seen_parents.emplace(parent_loc_idx);
  }

  return mapping;
}

}  // namespace motis::transfers::restrictions
