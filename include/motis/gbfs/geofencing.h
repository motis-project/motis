#pragma once

#include <vector>
#include <string>

#include "tg.h"

#include "geo/latlng.h"

namespace motis::gbfs {

bool applies(std::vector<std::string> const& rule_vehicle_type_ids,
             std::vector<std::string> const& segment_vehicle_type_ids);
bool multipoly_contains_point(tg_geom const* geom, geo::latlng const& pos);

}  // namespace motis::gbfs
