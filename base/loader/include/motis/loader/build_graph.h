#pragma once

#include <ctime>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/loader/loader_options.h"

namespace motis::loader {

struct Schedule;  // NOLINT

schedule_ptr build_graph(std::vector<Schedule const*> const&,
                         loader_options const&);

}  // namespace motis::loader
