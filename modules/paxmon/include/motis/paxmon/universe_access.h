#pragma once

#include <memory>

#include "motis/core/schedule/schedule.h"
#include "motis/module/ctx_data.h"
#include "motis/module/locked_resources.h"

#include "motis/paxmon/universe.h"
#include "motis/paxmon/universe_info.h"

namespace motis::paxmon {

struct universe_access {
  motis::module::locked_resources access_mutex_;
  schedule const& sched_;
  universe& uv_;
  std::shared_ptr<universe_info> uv_info_;
};

}  // namespace motis::paxmon
