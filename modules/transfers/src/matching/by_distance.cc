#include "motis/transfers/matching/by_distance.h"

#include <cstddef>
#include <limits>
#include <vector>

#include "motis/transfers/platform/platform.h"
#include "motis/transfers/types.h"

#include "nigiri/types.h"

#include "utl/progress_tracker.h"

namespace n = ::nigiri;

namespace motis::transfers {

matching_results distance_matcher::matching() {
  auto matches = matching_results{};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(
      data_.locations_to_match_.ids_.size());

  for (auto i = std::size_t{0U}; i < data_.locations_to_match_.names_.size();
       ++i) {
    progress_tracker->increment();
    auto nloc = data_.locations_to_match_.get(n::location_idx_t{i});

    if (data_.already_matched_nloc_keys_.count(to_key(nloc.pos_)) == 1) {
      continue;
    }

    // match location and platform: match to nearest platform
    auto [has_match, match] = match_by_distance(nloc);

    if (!has_match) {
      continue;
    }

    matches.emplace_back(match);
  }

  return matches;
};

std::pair<bool, matching_result> distance_matcher::match_by_distance(
    n::location const& nloc) {
  auto match = matching_result{};
  match.nloc_pos_ = nloc.pos_;

  auto has_match = false;
  auto shortest_dist = std::numeric_limits<double>::max();

  auto pfs_in_rad_with_dist = std::vector<std::pair<double, platform>>{};

  if (data_.has_update_state_pf_idx_) {
    auto update_pfs_in_rad_with_dist =
        data_.update_state_pf_idx_.get_platforms_in_radius_with_distance_info(
            nloc.pos_, options_.max_matching_dist_);
    pfs_in_rad_with_dist.insert(pfs_in_rad_with_dist.end(),
                                update_pfs_in_rad_with_dist.begin(),
                                update_pfs_in_rad_with_dist.end());
  }

  auto const old_pfs_in_rad_with_dist =
      data_.old_state_pf_idx_.get_platforms_in_radius_with_distance_info(
          nloc.pos_, options_.max_matching_dist_);
  pfs_in_rad_with_dist.insert(pfs_in_rad_with_dist.end(),
                              old_pfs_in_rad_with_dist.begin(),
                              old_pfs_in_rad_with_dist.end());

  for (auto [dist, pf] : pfs_in_rad_with_dist) {
    // only match bus stops with a distance of upt to a certain distance
    // (options)
    if (pf.is_bus_stop_ && dist > options_.max_bus_stop_matching_dist_) {
      continue;
    }

    if (dist < shortest_dist) {
      match.pf_ = pf;
      shortest_dist = dist;
      has_match = true;
    }
  }

  return {has_match, match};
}

}  // namespace motis::transfers
