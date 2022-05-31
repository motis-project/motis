#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace motis::paxmon::settings {

struct journey_input_settings {
  std::string journey_timezone_;
  std::string journey_match_log_file_{};
  std::uint16_t match_tolerance_{0};

  bool split_groups_{false};
  double split_groups_size_mean_{1.5};
  double split_groups_size_stddev_{3.0};
  unsigned split_groups_seed_{0};
};

}  // namespace motis::paxmon::settings
