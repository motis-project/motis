#pragma once

#include <string>
#include <vector>

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/types.h"

#include "nigiri/types.h"

namespace motis::footpaths {

struct transfer_request_keys {
  string from_pf_key_;
  string from_nloc_key_;

  strings to_pf_keys_;
  strings to_nloc_keys_;

  string profile_;
};
using transfer_requests_keys = std::vector<transfer_request_keys>;

struct transfer_request {
  platform transfer_start_;
  string from_nloc_key;

  platforms transfer_targets_;
  std::vector<string> to_nloc_keys;

  std::string profile_name;
};
using transfer_requests = std::vector<transfer_request>;

struct transfer_info {
  nigiri::duration_t duration_{};
  double distance_{};
};
using transfer_infos = std::vector<transfer_info>;

struct transfer_result {
  string from_nloc_key;
  string to_nloc_key;
  string profile_;

  transfer_info info_;
};
using transfer_results = std::vector<transfer_result>;

}  // namespace motis::footpaths
