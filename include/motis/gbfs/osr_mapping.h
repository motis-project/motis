#pragma once

#include "motis/fwd.h"

namespace motis::gbfs {

void map_geofencing_zones(gbfs_provider&, osr::ways const&, osr::lookup const&);
void map_stations(gbfs_provider&, osr::ways const&, osr::lookup const&);
void map_vehicles(gbfs_provider&, osr::ways const&, osr::lookup const&);

}  // namespace motis::gbfs
