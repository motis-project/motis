#pragma once

#include <string>

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfer_updates.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

struct state {
  std::unique_ptr<platforms_index> pfs_idx_;
  hash_map<std::string /* nloc key */, platform> matches_;
  hash_map<std::string /* from::to::profile */, transfer_result>
      transfer_results_;
};

}  // namespace motis::footpaths
