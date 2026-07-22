#include "motis/gbfs/geofencing.h"

#include "utl/helpers/algorithm.h"

#include "tg.h"

namespace motis::gbfs {

bool applies(std::vector<vehicle_type_idx_t> const& rule_vehicle_type_idxs,
             std::vector<vehicle_type_idx_t> const& segment_vehicle_type_idxs) {
  return rule_vehicle_type_idxs.empty() ||
         utl::all_of(segment_vehicle_type_idxs, [&](auto const& idx) {
           return utl::find(rule_vehicle_type_idxs, idx) !=
                  end(rule_vehicle_type_idxs);
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

geofencing_restrictions get_default_restrictions(
    gbfs_provider const& provider, provider_products const& product) {
  auto restrictions = provider.default_restrictions_;
  for (auto const& rule : provider.geofencing_zones_.global_rules_) {
    if (!applies(rule.vehicle_type_idxs_, product.vehicle_types_)) {
      continue;
    }
    restrictions.ride_start_allowed_ = rule.ride_start_allowed_;
    restrictions.ride_end_allowed_ = rule.ride_end_allowed_;
    restrictions.ride_through_allowed_ = rule.ride_through_allowed_;
    restrictions.station_parking_ = rule.station_parking_;
    break;
  }

  if ((product.return_constraint_ == return_constraint::kAnyStation ||
       product.return_constraint_ == return_constraint::kRoundtripStation) &&
      (product.known_return_constraint_ ||
       provider.geofencing_zones_.zones_.empty()) &&
      !restrictions.station_parking_.has_value()) {
    restrictions.station_parking_ = true;
  }
  return restrictions;
}

geofencing_restrictions get_restrictions(gbfs_provider const& provider,
                                         provider_products const& product,
                                         geo::latlng const& pos) {
  auto restrictions = get_default_restrictions(provider, product);
  for (auto const& zone : provider.geofencing_zones_.zones_) {
    auto const box = zone.bounding_box();
    if (pos.lat_ < box.min_.lat_ || pos.lat_ > box.max_.lat_ ||
        pos.lng_ < box.min_.lng_ || pos.lng_ > box.max_.lng_) {
      continue;
    }
    if (!multipoly_contains_point(zone.geom_.get(), pos)) {
      continue;
    }
    for (auto const& rule : zone.rules_) {
      if (!applies(rule.vehicle_type_idxs_, product.vehicle_types_)) {
        continue;
      }
      restrictions.ride_start_allowed_ = rule.ride_start_allowed_;
      restrictions.ride_end_allowed_ = rule.ride_end_allowed_;
      restrictions.ride_through_allowed_ = rule.ride_through_allowed_;
      if (rule.station_parking_.has_value()) {
        restrictions.station_parking_ = rule.station_parking_;
      }
      return restrictions;
    }
  }
  return restrictions;
}

bool allows_free_floating_return_at(gbfs_provider const& provider,
                                    provider_products const& product,
                                    geo::latlng const& pos,
                                    bool const ignore_return_constraints) {
  if (ignore_return_constraints) {
    return true;
  }
  auto const restrictions = get_restrictions(provider, product, pos);
  return restrictions.ride_end_allowed_ &&
         !restrictions.station_parking_.value_or(false);
}

bool vehicle_is_rentable(gbfs_provider const& provider,
                         provider_products const& product,
                         vehicle_status const& vehicle) {
  if (vehicle.is_disabled_ || vehicle.is_reserved_ ||
      !product.includes_vehicle_type(vehicle.vehicle_type_idx_)) {
    return false;
  }
  auto const restrictions = provider.geofencing_zones_.get_restrictions(
      vehicle.pos_, vehicle.vehicle_type_idx_,
      get_default_restrictions(provider, product));
  return restrictions.ride_start_allowed_ && restrictions.ride_through_allowed_;
}

geofencing_restrictions geofencing_zones::get_restrictions(
    geo::latlng const& pos,
    vehicle_type_idx_t const vehicle_type_idx,
    geofencing_restrictions const& default_restrictions) const {
  auto const check_vehicle_type =
      vehicle_type_idx != vehicle_type_idx_t::invalid();
  for (auto const& z : zones_) {
    if (multipoly_contains_point(z.geom_.get(), pos)) {
      for (auto const& r : z.rules_) {
        if (check_vehicle_type && !r.vehicle_type_idxs_.empty() &&
            utl::find(r.vehicle_type_idxs_, vehicle_type_idx) ==
                end(r.vehicle_type_idxs_)) {
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
