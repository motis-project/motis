#pragma once

#include "motis/paxmon/error.h"
#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/universe_access.h"

namespace motis::paxmon {

inline universe_access get_universe_and_schedule(
    paxmon_data& data, universe_id const id = 0,
    ctx::access_t const universe_access = ctx::access_t::READ,
    ctx::access_t const schedule_access = ctx::access_t::READ) {
  return data.multiverse_.get(id, universe_access, schedule_access);
}

}  // namespace motis::paxmon
