#pragma once

#include <cstdint>

namespace motis::paxmon {

struct additional_group {
  std::uint16_t passengers_{};
  float probability_{};
};

}  // namespace motis::paxmon
