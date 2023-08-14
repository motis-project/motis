#include "motis/footpaths/transfer_requests.h"

#include "motis/footpaths/types.h"

namespace motis::footpaths {

transfer_requests generate_new_all_reachable_pairs_requests(
    state const& old_state, state const& update_state,
    std::map<std::string, ppr::profile_info> const& profiles) {
  auto result = transfer_requests{};

  auto const all_pairs_trs = [&](state const& from_state, state const& to_state,
                                 std::string const& profile) {
    auto from_to_trs = transfer_requests{};
    auto const& pi = profiles.at(profile);
    auto prf_dist = pi.profile_.walking_speed_ * pi.profile_.duration_limit_;

    for (auto i = std::size_t{0}; i < from_state.matched_pfs_idx_->size();
         ++i) {

      auto start = from_state.matched_pfs_idx_->get_platform(i);
      auto target_ids =
          to_state.matched_pfs_idx_->valid_in_radius(start, prf_dist);

      if (target_ids.empty()) {
        continue;
      }

      auto tmp = transfer_request{};
      auto target_pfs = platforms{};
      auto target_nloc_keys = std::vector<string>{};

      for (auto i : target_ids) {
        target_pfs.emplace_back(to_state.matched_pfs_idx_->get_platform(i));
      }

      for (auto i : target_ids) {
        target_nloc_keys.emplace_back(to_state.nloc_keys[i]);
      }

      tmp.transfer_start_ = start;
      tmp.from_nloc_key = to_state.nloc_keys[i];
      tmp.transfer_targets_ = target_pfs;
      tmp.to_nloc_keys = target_nloc_keys;
      tmp.profile_name = profile;

      from_to_trs.emplace_back(tmp);
    }

    return from_to_trs;
  };

  // new possible transfers: 1 -> 2, 2 -> 1, 2 -> 2
  for (auto const& [prf_name, prf_info] : profiles) {
    // new transfers from old to update (1 -> 2)
    auto trs12 = all_pairs_trs(old_state, update_state, prf_name);
    // new transfers from update to old (2 -> 1)
    auto trs21 = all_pairs_trs(update_state, old_state, prf_name);
    // new transfers from update to update (2 -> 2)
    auto trs22 = all_pairs_trs(update_state, update_state, prf_name);

    result.insert(result.end(), trs12.begin(), trs12.end());
    result.insert(result.end(), trs21.begin(), trs21.end());
    result.insert(result.end(), trs22.begin(), trs22.end());
  }

  return result;
}

}  // namespace motis::footpaths
