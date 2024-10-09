#pragma once

#include <string>

#include "nigiri/types.h"
#include "motis/fwd.h"

namespace motis {

std::string get_service_date(nigiri::timetable const& tt,
                             nigiri::transport,
                             nigiri::stop_idx_t);

}  // namespace motis
