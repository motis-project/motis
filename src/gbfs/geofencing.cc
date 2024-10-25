#include "motis/gbfs/data.h"

#include "tg.h"

namespace motis::gbfs {

bool multipoly_contains_point(tg_geom const* geom, geo::latlng const& pos) {
  auto const n_polys = tg_geom_num_polys(geom);
  for (auto i = 0; i < n_polys; ++i) {
    auto const* poly = tg_geom_poly_at(geom, i);
    if (tg_geom_intersects_xy(reinterpret_cast<tg_geom const*>(poly), pos.lng(),
                              pos.lat())) {
      return true;
    }
  }
  return false;
}

geofencing_restrictions geofencing_zones::get_restrictions(
    geo::latlng const& pos,
    geofencing_restrictions const& default_restrictions) {
  for (auto const& z : zones_) {
    if (multipoly_contains_point(z.geom_.get(), pos)) {
      // vehicle_type_ids currently ignored, using first rule
      if (!z.rules_.empty()) {
        auto const& r = z.rules_.front();
        return geofencing_restrictions{
            .ride_start_allowed_ = r.ride_start_allowed_,
            .ride_end_allowed_ = r.ride_end_allowed_,
            .ride_through_allowed_ = r.ride_through_allowed_,
            .station_parking_ = r.station_parking_};
      }
    }
  }
  return default_restrictions;
}

}  // namespace motis::gbfs
