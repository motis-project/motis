#pragma once

#include <string>

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

struct state {
  std::unique_ptr<platforms_index> pfs_idx_;
  std::unique_ptr<platforms_index> matched_pfs_idx_;
  std::vector<std::string> nloc_keys;
  hash_map<std::string /* nloc key */, platform> matches_;
  transfer_results transfer_results_;
};

}  // namespace motis::footpaths
