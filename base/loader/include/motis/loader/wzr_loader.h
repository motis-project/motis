#pragma once

#include <string>

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/core/schedule/category.h"
#include "motis/core/schedule/waiting_time_rules.h"

#include "motis/core/schedule/schedule.h"

namespace motis::loader {

waiting_time_rules load_waiting_time_rules(
    std::string const& wzr_classes_path, std::string const& wzr_matrix_path,
    mcd::vector<mcd::unique_ptr<category>> const& category_ptrs);

void calc_waits_for(schedule& sched, duration planned_transfer_delta);

}  // namespace motis::loader
