#pragma once

#include "cista/memory_holder.h"

#include "motis/core/schedule/schedule.h"

namespace motis {

schedule_ptr read_graph(std::string const& path, cista::memory_holder&,
                        bool read_mmap);

void write_graph(std::string const& path, schedule const&);

schedule_data copy_graph(schedule const& sched);

}  // namespace motis
