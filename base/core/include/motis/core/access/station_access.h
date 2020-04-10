#pragma once

#include <string_view>

#include "motis/core/schedule/schedule.h"

namespace motis {

station* find_station(schedule const&, std::string_view eva_nr);
station* get_station(schedule const&, std::string_view eva_nr);
station_node* get_station_node(schedule const&, std::string_view eva_nr);
station_node* find_station_node(schedule const&, std::string_view eva_nr);

}  // namespace motis
