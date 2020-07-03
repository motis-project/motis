#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "motis/core/schedule/schedule.h"

namespace motis::paxmon::util {

inline std::optional<std::uint32_t> get_station_idx(schedule const& sched,
                                                    std::string_view id) {
  if (id.empty()) {
    return {};
  } else if (auto const st = find_station(sched, id); st != nullptr) {
    return st->index_;
  } else if (auto const it = sched.ds100_to_station_.find(id);
             it != end(sched.ds100_to_station_)) {
    return it->second->index_;
  }
  return {};
}

}  // namespace motis::paxmon::util
