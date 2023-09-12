#include "motis/transfers/matching.h"

#include "utl/progress_tracker.h"

#include <limits>

namespace n = ::nigiri;

namespace motis::transfers {

matching_results match_locations_and_platforms(matching_data const& data,
                                               matching_options const& opts) {
  auto matches = matching_results{};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(
      data.locations_to_match_.ids_.size());

  for (auto i = 0U; i < data.locations_to_match_.ids_.size(); ++i) {
    progress_tracker->increment();
    auto nloc = data.locations_to_match_.get(n::location_idx_t{i});

    if (data.already_matched_nloc_keys_.count(to_key(nloc.pos_)) == 1) {
      continue;
    }

    // match location and platform using exact name match
    auto [has_match_up, match_res] = match_by_distance(nloc, data, opts);

    if (!has_match_up) {
      continue;
    }

    matches.emplace_back(match_res);
  }

  return matches;
}

// -- match functions --
std::pair<bool, matching_result> match_by_distance(
    n::location const& nloc, matching_data const& data,
    matching_options const& opts) {
  auto mr = matching_result{};
  mr.nloc_pos_ = nloc.pos_;

  auto matched{false};
  auto shortest_dist{std::numeric_limits<double>::max()};

  auto pfs = std::vector<std::pair<double, motis::transfers::platform>>{};
  // use update_state and old_state
  if (data.has_update_state_pf_idx_) {
    auto new_pfs =
        data.update_state_pf_idx_.get_platforms_in_radius_with_distance_info(
            nloc.pos_, opts.max_matching_dist_);
    pfs.insert(pfs.end(), new_pfs.begin(), new_pfs.end());
  }
  auto const old_pfs =
      data.old_state_pf_idx_.get_platforms_in_radius_with_distance_info(
          nloc.pos_, opts.max_matching_dist_);
  pfs.insert(pfs.end(), old_pfs.begin(), old_pfs.end());

  for (auto [dist, pf] : pfs) {
    // only match bus stops with a distance of up to a certain distance
    if (pf.is_bus_stop_ && dist > opts.max_bus_stop_matching_dist_) {
      continue;
    }

    if (dist < shortest_dist) {
      mr.pf_ = pf;
      shortest_dist = dist;
      matched = true;
    }
  }

  return {matched, mr};
}

}  // namespace motis::transfers
