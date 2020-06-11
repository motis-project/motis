#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/tripbased/data.h"

namespace motis::tripbased {

std::unique_ptr<tb_data> build_data(schedule const& sched);

std::unique_ptr<tb_data> load_data(schedule const& sched,
                                   std::string const& filename);

void update_data_file(schedule const& sched, std::string const& filename,
                      bool force_update);

}  // namespace motis::tripbased
