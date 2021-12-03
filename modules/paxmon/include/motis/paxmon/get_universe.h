#pragma once

#include "motis/paxmon/error.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

inline universe& get_primary_universe(paxmon_data& data) {
  return data.multiverse_.primary();
}

inline universe& get_universe(paxmon_data& data, universe_id const id) {
  if (auto uv = data.multiverse_.try_get(id); uv) {
    return *uv.value();
  } else {
    throw std::system_error{error::universe_not_found};
  }
}

}  // namespace motis::paxmon
