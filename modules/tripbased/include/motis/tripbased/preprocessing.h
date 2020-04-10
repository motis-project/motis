#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/tripbased/data.h"

namespace motis::tripbased {

std::unique_ptr<tb_data> build_data(schedule const& sched);

std::unique_ptr<tb_data> load_or_build_data(schedule const& sched,
                                            std::string const& filename);

}  // namespace motis::tripbased
