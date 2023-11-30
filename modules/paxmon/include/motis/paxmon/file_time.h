#pragma once

#include <chrono>
#include <filesystem>

#include "motis/core/common/unixtime.h"

namespace motis::paxmon {

inline unixtime get_last_modified_time(std::filesystem::path const& path) {
  auto const file_time = std::filesystem::last_write_time(path);

#ifdef _MSC_VER
  auto const sys_time =
      std::chrono::clock_cast<std::chrono::system_clock>(file_time);
#else
  auto const sys_time =
      std::chrono::time_point_cast<std::chrono::system_clock::duration>(
          file_time - std::chrono::file_clock::now() +
          std::chrono::system_clock::now());
#endif

  return static_cast<unixtime>(
      std::chrono::time_point_cast<std::chrono::seconds>(sys_time)
          .time_since_epoch()
          .count());
}

}  // namespace motis::paxmon
