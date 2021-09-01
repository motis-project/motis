#pragma once

#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

inline bool is_local_station(Station const* station) {
  return station->id()->c_str()[0] == '0';
}

}  // namespace motis::loader
