#pragma once

#include "tg.h"

#include "geo/latlng.h"

namespace motis::gbfs {

bool multipoly_contains_point(tg_geom const* geom, geo::latlng const& pos);

}  // namespace motis::gbfs
