#pragma once

#include <string>

#include "boost/json.hpp"

#include "motis/gbfs/data.h"
#include "motis/types.h"

namespace motis::gbfs {

hash_map<std::string, std::string> parse_discovery(
    boost::json::value const& root);

std::optional<return_constraint> parse_return_constraint(std::string_view s);

void load_system_information(gbfs_provider&, boost::json::value const& root);
void load_station_information(gbfs_provider&, boost::json::value const& root);
void load_station_status(gbfs_provider&, boost::json::value const& root);
void load_vehicle_types(gbfs_provider&, boost::json::value const& root);
void load_vehicle_status(gbfs_provider&, boost::json::value const& root);
void load_geofencing_zones(gbfs_provider&, boost::json::value const& root);

}  // namespace motis::gbfs
