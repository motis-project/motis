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
 * @param max_walk_duration the maximmum walking duration to identify platforms
 * in range of a given position
 *
 */
std::vector<transfer_requests> build_transfer_requests(
    platforms* pf, std::map<std::string, ppr::profile_info> const& profiles,
    int const max_walk_duration);

}  // namespace motis::footpaths
