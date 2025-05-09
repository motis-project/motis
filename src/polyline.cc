#include "motis/polyline.h"

#include "geo/polyline_format.h"

namespace motis {

template <std::int64_t Precision>
api::EncodedPolyline to_polyline(geo::polyline const& polyline) {
  return {geo::encode_polyline<Precision>(polyline), Precision,
          static_cast<std::int64_t>(polyline.size())};
}

template api::EncodedPolyline to_polyline<5>(geo::polyline const&);
template api::EncodedPolyline to_polyline<6>(geo::polyline const&);
template api::EncodedPolyline to_polyline<7>(geo::polyline const&);

}  // namespace motis