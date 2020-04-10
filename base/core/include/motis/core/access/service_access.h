#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/error.h"

namespace motis {

uint32_t output_train_nr(uint32_t train_nr, uint32_t original_train_nr);

std::string get_service_name(schedule const& sched,
                             connection_info const* info);

}  // namespace motis
