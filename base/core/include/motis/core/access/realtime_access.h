#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis {

time get_schedule_time(schedule const&, ev_key const&);

time get_schedule_time(schedule const&, edge const* route_edge, lcon_idx_t,
                       event_type, day_idx_t);

time get_schedule_time(schedule const&, edge const* route_edge,
                       light_connection const*, event_type, day_idx_t);

duration_t get_delay(schedule const&, ev_key const&);

delay_info get_delay_info(schedule const&, node const* route_node,
                          light_connection const*, event_type);

delay_info get_delay_info(schedule const&, edge const* route_edge,
                          light_connection const*, event_type);

delay_info get_delay_info(schedule const&, ev_key const&);

ev_key const& get_current_ev_key(schedule const&, ev_key const&);

ev_key const& get_orig_ev_key(schedule const&, ev_key const&);

uint16_t get_schedule_track(schedule const&, ev_key const&);

}  // namespace motis
