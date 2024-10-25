#pragma once

#include "motis/fwd.h"

namespace motis::gbfs {

void map_geofencing_zones(osr::ways const&, osr::lookup const&, gbfs_provider&);
void map_stations(osr::ways const&, osr::lookup const&, gbfs_provider&);
void map_vehicles(osr::ways const&, osr::lookup const&, gbfs_provider&);

}  // namespace motis::gbfs
