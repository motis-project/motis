#include "motis/footpaths/transfer_requests.h"

namespace motis::footpaths {

transfer_requests generate_new_all_reachable_pairs_requests(
    state const& old_state, state const& update_state,
    std::map<std::string, ppr::profile_info> const& profiles) {
  auto result = transfer_requests{};

  auto const all_pairs_trs = [&](platforms_index* from, platforms_index* to,
                                 std::string const& profile) {
    auto from_to_trs = transfer_requests{};
    auto const& pi = profiles.at(profile);
    auto prf_dist = pi.profile_.walking_speed_ * pi.profile_.duration_limit_;

    for (auto i = std::size_t{0}; i < from->size(); ++i) {
      auto tmp = transfer_request{};

      auto start = from->get_platform(i);
      auto targets = to->valid_in_radius(start, prf_dist);

      if (targets.empty()) {
        continue;
      }

      tmp.transfer_start_ = start;
      tmp.transfer_targets_ = targets;
      tmp.profile_name = profile;

      from_to_trs.emplace_back(tmp);
    }

    return from_to_trs;
  };

  // new possible transfers: 1 -> 2, 2 -> 1, 2 -> 2
  for (auto const& [prf_name, prf_info] : profiles) {
    // new transfers from old to update (1 -> 2)
    auto trs12 = all_pairs_trs(old_state.matched_pfs_idx_.get(),
                               update_state.matched_pfs_idx_.get(), prf_name);
    // new transfers from update to old (2 -> 1)
    auto trs21 = all_pairs_trs(update_state.matched_pfs_idx_.get(),
                               old_state.matched_pfs_idx_.get(), prf_name);
    // new transfers from update to update (2 -> 2)
    auto trs22 = all_pairs_trs(update_state.matched_pfs_idx_.get(),
                               update_state.matched_pfs_idx_.get(), prf_name);

    result.insert(result.end(), trs12.begin(), trs12.end());
    result.insert(result.end(), trs21.begin(), trs21.end());
    result.insert(result.end(), trs22.begin(), trs22.end());
  }

  return result;
}

}  // namespace motis::footpaths
