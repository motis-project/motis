#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis::access {

connection_info const& get_connection_info(schedule const&,
                                           generic_light_connection const&,
                                           trip_info const*);

}  // namespace motis::access
