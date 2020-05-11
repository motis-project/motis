#pragma once

#include "cista/memory_holder.h"

#include "motis/core/schedule/schedule.h"

#include "motis/loader/loader_options.h"
#include "motis/loader/parser.h"

namespace motis::loader {

std::vector<std::unique_ptr<format_parser>> parsers();

schedule_ptr load_schedule(loader_options const&,
                           cista::memory_holder& schedule_buf);

schedule_ptr load_schedule(loader_options const&);

}  // namespace motis::loader
