#pragma once

#include "motis/footpaths/platforms.h"
#include "motis/ppr/profiles.h"

#include <vector>

namespace motis::footpaths {

struct transfer_requests {
  platform_info* transfer_start_;
  std::vector<platform_info*> transfer_targets_;
  std::string profile_name;
};

/**
 * Creates all valid transfer requests for a given list of platforms.
 *
 * @param pf the struct containing the list of platforms
 * @param profile_info a list of ppr profiles for the creation of transfer
 * requests
 *
 * @returns a list of transfer requests (start to all destinations) for the
 * given profile
 *
 */
std::vector<transfer_requests> build_transfer_requests(
    platforms* pf, std::map<std::string, ppr::profile_info> const& profiles);

}  // namespace motis::footpaths
