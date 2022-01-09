#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis {

uint32_t get_track_string_idx(schedule const&, generic_light_connection const&,
                              event_type);

mcd::string const& get_track_name(schedule const&,
                                  generic_light_connection const&, event_type);

}  // namespace motis
