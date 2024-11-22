#include "motis/gbfs/data.h"

#include "utl/helpers/algorithm.h"

#include "tg.h"

namespace motis::gbfs {

bool applies(std::vector<std::string> const& rule_vehicle_type_ids,
             std::vector<std::string> const& segment_vehicle_type_ids) {
  return rule_vehicle_type_ids.empty() ||
         utl::all_of(segment_vehicle_type_ids, [&](auto const& id) {
           return utl::find(rule_vehicle_type_ids, id) !=
                  end(rule_vehicle_type_ids);
         });
}

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
    std::string const& vehicle_type_id,
    geofencing_restrictions const& default_restrictions) const {
  auto const check_vehicle_type = !vehicle_type_id.empty();
  for (auto const& z : zones_) {
    if (multipoly_contains_point(z.geom_.get(), pos)) {
      for (auto const& r : z.rules_) {
        if (check_vehicle_type && !r.vehicle_type_ids_.empty() &&
            utl::find(r.vehicle_type_ids_, vehicle_type_id) ==
                end(r.vehicle_type_ids_)) {
          continue;
        }
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
