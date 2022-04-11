#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "cista/reflection/comparable.h"

namespace motis::gbfs {

struct urls {
  CISTA_COMPARABLE()
  std::string lang_;
  std::optional<std::string> free_bike_url_;
  std::optional<std::string> station_info_url_;
  std::optional<std::string> station_status_url_;
};

std::vector<urls> read_system_status(std::string_view);

}  // namespace motis::gbfs
