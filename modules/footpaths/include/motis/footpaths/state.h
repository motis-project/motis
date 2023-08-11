#pragma once

#include <string>

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

struct state {
  std::unique_ptr<platforms_index> pfs_idx_;
  hash_map<std::string /* nloc key */, platform> matches_;
};

}  // namespace motis::footpaths
