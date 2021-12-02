#pragma once

#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/schedule.h"

namespace motis {

node* get_or_add_platform_node(schedule& sched, station_node* station_node,
                               uint16_t platform);

node* add_platform_enter_edge(schedule& sched, node* route_node,
                              station_node* station_node,
                              int32_t platform_transfer_time,
                              uint16_t platform);

node* add_platform_exit_edge(schedule& sched, node* route_node,
                             station_node* station_node,
                             int32_t platform_transfer_time, uint16_t platform);

}  // namespace motis
