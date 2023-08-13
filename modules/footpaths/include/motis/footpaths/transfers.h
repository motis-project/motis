#pragma once

#include <vector>

#include "geo/latlng.h"

#include "motis/footpaths/platforms.h"

namespace motis::footpaths {

struct transfer_request {
  platform transfer_start_;
  std::vector<platform> transfer_targets_;
  std::string profile_name;
};
using transfer_requests = std::vector<transfer_request>;

struct transfer_info {
  double duration_;
};

struct transfer_result {
  geo::latlng from_;
  geo::latlng to_;
  cista::raw::string profile_;

  transfer_info info_;
};
using transfer_results = std::vector<transfer_result>;

}  // namespace motis::footpaths
