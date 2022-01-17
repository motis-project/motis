#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/ctx_data.h"
#include "motis/module/locked_resources.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct universe_access {
  motis::module::locked_resources access_mutex_;
  schedule const& sched_;
  universe& uv_;
};

}  // namespace motis::paxmon
