#pragma once

#include <vector>

#include "motis/footpaths/platforms.h"
#include "motis/ppr/profiles.h"

namespace motis::footpaths {

struct transfer_requests {
  platform* transfer_start_;
  std::vector<platform*> transfer_targets_;
  std::string profile_name;
};

std::vector<transfer_requests> build_transfer_requests(
    platforms_index*, std::map<std::string, ppr::profile_info> const&);

}  // namespace motis::footpaths
