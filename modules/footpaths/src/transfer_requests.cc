#include "motis/footpaths/transfer_requests.h"

namespace motis::footpaths {

std::vector<transfer_requests> build_transfer_requests(
    platforms* pf, std::map<std::string, ppr::profile_info> const& profiles,
    int const max_walk_duration) {
  std::vector<transfer_requests> result{};

  // every platform (or station) can be a start for a transfer: set every
  // platform as start
  for (auto& platform : pf->platforms_) {
    // create transfer requests for platforms with a valid location idx only
    if (platform.idx_ == nigiri::location_idx_t::invalid()) {
      continue;
    }

    std::vector<platform_info*> transfer_targets{};
    // different profiles result in different transfer_targets: determine for
    // each profile the reachable platforms
    for (auto& profile : profiles) {
      // remark: profile {profile_name -> profile_info}
      // get all valid platforms in radius of current platform
      auto valid_platforms_in_radius = pf->get_valid_platforms_in_radius(
          &platform,
          profile.second.profile_.walking_speed_ * max_walk_duration);
      transfer_targets.insert(transfer_targets.end(),
                              valid_platforms_in_radius.begin(),
                              valid_platforms_in_radius.end());

      // donot create a transfer request if no valid transfer could be found
      if (transfer_targets.empty()) {
        continue;
      }

      transfer_requests tmp{};
      tmp.transfer_start_ = &platform;
      tmp.transfer_targets_ = transfer_targets;
      tmp.profile_name = profile.first;

      result.emplace_back(tmp);
    }
  }

  return result;
}

}  // namespace motis::footpaths
