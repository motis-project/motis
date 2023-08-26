#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "fmt/core.h"

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/types.h"

#include "nigiri/types.h"

namespace motis::footpaths {

struct transfer_request_keys {
  CISTA_COMPARABLE();
  nlocation_key_t from_nloc_key_;
  vector<nlocation_key_t> to_nloc_keys_;
  profile_key_t profile_;
};
using transfer_requests_keys = std::vector<transfer_request_keys>;

struct transfer_request {
  platform transfer_start_;
  nlocation_key_t from_nloc_key_;

  platforms transfer_targets_;
  vector<nlocation_key_t> to_nloc_keys_;

  profile_key_t profile_;
};
using transfer_requests = std::vector<transfer_request>;

struct transfer_info {
  CISTA_COMPARABLE();
  nigiri::duration_t duration_{};
  double distance_{};
};
using transfer_infos = std::vector<transfer_info>;

struct transfer_result {
  CISTA_COMPARABLE();
  nlocation_key_t from_nloc_key_;
  vector<nlocation_key_t> to_nloc_keys_;
  profile_key_t profile_;

  vector<transfer_info> infos_;
};
using transfer_results = std::vector<transfer_result>;

transfer_request_keys merge(transfer_request_keys const&,
                            transfer_request_keys const&);
transfer_result merge(transfer_result const&, transfer_result const&);

// ostreams
std::ostream& operator<<(std::ostream&, transfer_info const&);
std::ostream& operator<<(std::ostream&, transfer_result const&);
std::ostream& operator<<(std::ostream&, transfer_request const&);
std::ostream& operator<<(std::ostream&, transfer_request_keys const&);

}  // namespace motis::footpaths
