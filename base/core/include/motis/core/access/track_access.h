#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis {

mcd::string const& get_track_name(schedule const&, uint32_t track_idx,
                                  day_idx_t);

uint32_t get_track_string_idx(schedule const& sched, uint32_t const track_idx,
                              day_idx_t const day_idx);

}  // namespace motis
