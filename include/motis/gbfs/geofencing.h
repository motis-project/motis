#pragma once

#include <string>
#include <vector>

#include "tg.h"

#include "geo/latlng.h"

#include "motis/gbfs/data.h"

namespace motis::gbfs {

bool applies(std::vector<vehicle_type_idx_t> const& rule_vehicle_type_idxs,
             std::vector<vehicle_type_idx_t> const& segment_vehicle_type_idxs);
bool multipoly_contains_point(tg_geom const* geom, geo::latlng const& pos);
geofencing_restrictions get_default_restrictions(gbfs_provider const&,
                                                 provider_products const&);
bool vehicle_is_rentable(gbfs_provider const&,
                         provider_products const&,
                         vehicle_status const&);

}  // namespace motis::gbfs
