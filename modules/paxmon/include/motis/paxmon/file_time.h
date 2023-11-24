#pragma once

#include <chrono>
#include <filesystem>

#include "motis/core/common/unixtime.h"

namespace motis::paxmon {

inline unixtime get_last_modified_time(std::filesystem::path const& path) {
  return static_cast<unixtime>(
      std::chrono::time_point_cast<std::chrono::seconds>(
          std::filesystem::last_write_time(path))
          .time_since_epoch()
          .count());
}

}  // namespace motis::paxmon
