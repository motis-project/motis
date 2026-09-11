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
    gbfs_provider const& provider,
    provider_products const& product,
    std::vector<rule> const& global_rules) {
  auto restrictions = provider.default_restrictions_;
  for (auto const& rule : global_rules) {
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

geofencing_restrictions get_default_restrictions(
    gbfs_provider const& provider, provider_products const& product) {
  auto global_rules = std::vector<rule>{};
  for (auto const& zone : provider.geofencing_zones_.zones_) {
    if (zone.is_global() && provider.geofencing_zones_.zones_.size() != 1U) {
      global_rules.insert(global_rules.begin(), zone.rules_.begin(),
                          zone.rules_.end());
    }
  }
  global_rules.insert(global_rules.end(),
                      provider.geofencing_zones_.global_rules_.begin(),
                      provider.geofencing_zones_.global_rules_.end());
  return get_default_restrictions(provider, product, global_rules);
}

geofencing_restrictions get_restrictions(gbfs_provider const& provider,
                                         provider_products const& product,
                                         geo::latlng const& pos) {
  auto const& zones = provider.geofencing_zones_;
  auto indices = std::vector<std::size_t>{};
  for (auto i = std::size_t{0}; i != zones.zones_.size(); ++i) {
    auto const& zone = zones.zones_[i];
    if (zone.is_global() && zones.zones_.size() != 1U) {
      continue;
    }
    if (zone.contains(pos)) {
      indices.push_back(i);
    }
  }
  return zones.get_restrictions(product.vehicle_types_,
                                get_default_restrictions(provider, product),
                                indices);
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
  auto const restrictions = get_restrictions(provider, product, vehicle.pos_);
  return restrictions.ride_start_allowed_ && restrictions.ride_through_allowed_;
}

geofencing_restrictions geofencing_zones::get_restrictions(
    std::vector<vehicle_type_idx_t> const& vehicle_types,
    geofencing_restrictions restrictions,
    std::span<std::size_t const> const zone_indices) const {
  for (auto const idx : zone_indices) {
    for (auto const& rule : zones_[idx].rules_) {
      if (!applies(rule.vehicle_type_idxs_, vehicle_types)) {
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

bool zone::has_exterior() const {
  if (!respect_winding_) {
    return false;
  }
  for (auto i = 0; i != tg_geom_num_polys(geom_.get()); ++i) {
    if (!tg_poly_clockwise(tg_geom_poly_at(geom_.get(), i))) {
      return true;
    }
  }
  return false;
}

bool zone::contains(geo::latlng const& pos) const {
  auto has_exterior = false;
  auto inside_exterior = false;
  for (auto i = 0; i != tg_geom_num_polys(geom_.get()); ++i) {
    auto const* poly = tg_geom_poly_at(geom_.get(), i);
    auto const inside = tg_geom_intersects_xy(
        reinterpret_cast<tg_geom const*>(poly), pos.lng(), pos.lat());
    if (respect_winding_ && !tg_poly_clockwise(poly)) {
      has_exterior = true;
      inside_exterior = inside_exterior || inside;
    } else if (inside) {
      return true;
    }
  }
  // Multiple operating areas describe the exterior of their union, rather
  // than each area prohibiting rides inside the other areas. Holes remain
  // part of the complement; clockwise components add interior restrictions.
  return has_exterior && !inside_exterior;
}

}  // namespace motis::gbfs
