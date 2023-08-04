#include "motis/footpaths/transfer_requests.h"

#include "motis/core/common/logging.h"

using namespace motis::logging;

namespace motis::footpaths {

std::vector<transfer_requests> build_transfer_requests(
    platforms_index* pfs_idx,
    std::map<std::string, ppr::profile_info> const& profiles) {
  u_int targets = 0, no_targets = 0;
  u_int stations = 0, tracks = 0;
  std::vector<transfer_requests> result{};

  // every platform (or station) can be a start for a transfer: set every
  // platform as start
  for (auto& pf : pfs_idx->platforms_) {
    // create transfer requests for platforms with a valid location idx only
    if (pf.info_.idx_ == nigiri::location_idx_t::invalid()) {
      continue;
    }
    // count stations and tracks
    pf.info_.osm_id_ == -1 ? ++stations : ++tracks;

    // different profiles result in different transfer_targets: determine for
    // each profile the reachable platforms
    for (auto const& profile : profiles) {
      std::vector<platform*> transfer_targets{};

      // remark: profile {profile_name -> profile_info}
      // get all valid platforms in radius of current platform
      auto valid_platforms_in_radius = pfs_idx->get_valid_platforms_in_radius(
          &pf, profile.second.profile_.walking_speed_ *
                   profile.second.profile_.duration_limit_);
      transfer_targets.insert(transfer_targets.end(),
                              valid_platforms_in_radius.begin(),
                              valid_platforms_in_radius.end());
      targets += valid_platforms_in_radius.size();

      // donot create a transfer request if no valid transfer could be found
      if (transfer_targets.empty()) {
        ++no_targets;
        continue;
      }

      transfer_requests tmp{};
      tmp.transfer_start_ = &pf;
      tmp.transfer_targets_ = transfer_targets;
      tmp.profile_name = profile.first;

      result.emplace_back(tmp);
    }
  }

  LOG(info) << "Generated " << result.size() << " transfer requests.";
  LOG(info) << "Found " << targets << " targets in total.";
  LOG(info) << "Identified "
            << (static_cast<double>(targets) /
                static_cast<double>(result.size()))
            << " targets per source.";
  LOG(info) << "Found " << no_targets
            << " (src, profile)-tuples w/o targets. (not included in transfer "
               "requests).";

  return result;
}

}  // namespace motis::footpaths
