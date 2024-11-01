#pragma once

#include "geo/polyline.h"

#include "motis-api/motis-api.h"

namespace motis {

template <std::int64_t Precision>
api::EncodedPolyline to_polyline(geo::polyline const& polyline);

}  // namespace motis