#pragma once

#include <memory>
#include <vector>

#include "motis/footpaths/platform/platform_index.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

enum class first_update { kNoUpdate, kProfiles, kTimetable, kOSM };
enum class routing_type { kNoRouting, kPartialRouting, kFullRouting };

struct state {
  std::unique_ptr<platform_index> pfs_idx_;
  std::unique_ptr<platform_index> matched_pfs_idx_;

  bool set_matched_pfs_idx_{false};
  bool set_pfs_idx_{false};

  // nloc_keys_.size() == matches_.size()
  vector<nlocation_key_t> nloc_keys_;  // matched nigiri location keys
  hash_map<nlocation_key_t /* nloc key */, platform>
      matches_;  // mapping matched nloc to pf
  transfer_requests_keys transfer_requests_keys_;
  transfer_results transfer_results_;
};

}  // namespace motis::footpaths
