#pragma once

#include <memory>
#include <vector>

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

enum class first_update { kNoUpdate, kProfiles, kTimetable, kOSM };
enum class routing_type { kNoRouting, kPartialRouting, kFullRouting };

struct state {
  std::unique_ptr<platforms_index> pfs_idx_;
  std::unique_ptr<platforms_index> matched_pfs_idx_;

  bool set_matched_pfs_idx_{false};
  bool set_pfs_idx_{false};

  std::vector<string> nloc_keys;
  hash_map<string /* nloc key */, platform> matches_;
  transfer_requests_keys transfer_requests_keys_;
  transfer_results transfer_results_;
};

}  // namespace motis::footpaths
