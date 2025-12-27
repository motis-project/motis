#include "motis/endpoints/map/rental.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <set>
#include <utility>
#include <vector>

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/to_vec.h"

#include "geo/box.h"
#include "geo/polyline.h"
#include "geo/polyline_format.h"

#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/mode.h"
#include "motis/parse_location.h"
#include "motis/place.h"

namespace json = boost::json;

namespace motis::ep {

api::rentals_response rental::operator()(
    boost::urls::url_view const& url) const {
  auto const parse_loc = [](std::string_view const sv) {
    return parse_location(sv);
  };
  auto const parse_place_pos = [&](std::string_view const sv) {
    auto const place = get_place(tt_, tags_, sv);
    return std::visit(
        utl::overloaded{[&](osr::location const& l) { return l.pos_; },
                        [&](tt_location const tt_l) {
                          return tt_->locations_.coordinates_.at(tt_l.l_);
                        }},
        place);
  };

  auto const query = api::rentals_params{url.params()};
  auto const min = query.min_.and_then(parse_loc);
  auto const max = query.max_.and_then(parse_loc);
  auto const point_pos = query.point_.transform(parse_place_pos);
  auto const point_radius = query.radius_;
  auto const filter_bbox = min.has_value() && max.has_value();
  auto const filter_point = point_pos.has_value() && point_radius.has_value();
  auto const filter_providers =
      query.providers_.has_value() && !query.providers_->empty();
  auto const filter_groups =
      query.providerGroups_.has_value() && !query.providerGroups_->empty();

  auto gbfs = gbfs_;
  auto res = api::rentals_response{};

  if (gbfs == nullptr) {
    return res;
  }

  auto const restrictions_to_api = [&](gbfs::geofencing_restrictions const& r) {
    return api::RentalZoneRestrictions{
        .vehicleTypeIdxs_ = {},
        .rideStartAllowed_ = r.ride_start_allowed_,
        .rideEndAllowed_ = r.ride_end_allowed_,
        .rideThroughAllowed_ = r.ride_through_allowed_,
        .stationParking_ = r.station_parking_};
  };

  auto const rule_to_api = [&](gbfs::rule const& r) {
    return api::RentalZoneRestrictions{
        .vehicleTypeIdxs_ =
            utl::to_vec(r.vehicle_type_idxs_,
                        [&](auto const vti) {
                          return static_cast<std::int64_t>(to_idx(vti));
                        }),
        .rideStartAllowed_ = r.ride_start_allowed_,
        .rideEndAllowed_ = r.ride_end_allowed_,
        .rideThroughAllowed_ = r.ride_through_allowed_,
        .stationParking_ = r.station_parking_};
  };

  auto const ring_to_api = [&](tg_ring const* ring) {
    auto enc = geo::polyline_encoder<6>{};
    auto const np = tg_ring_num_points(ring);
    for (auto i = 0; i != np; ++i) {
      auto const pt = tg_ring_point_at(ring, i);
      enc.push(geo::latlng{pt.y, pt.x});
    }
    return api::EncodedPolyline{
        .points_ = std::move(enc.buf_), .precision_ = 6, .length_ = np};
  };

  auto const multipoly_to_api = [&](tg_geom* const geom) {
    assert(tg_geom_typeof(geom) == TG_MULTIPOLYGON);
    auto mp = api::MultiPolygon{};
    for (auto i = 0; i != tg_geom_num_polys(geom); ++i) {
      auto const* poly = tg_geom_poly_at(geom, i);
      auto polylines = std::vector<api::EncodedPolyline>{};
      polylines.emplace_back(ring_to_api(tg_poly_exterior(poly)));
      for (int j = 0; j != tg_poly_num_holes(poly); ++j) {
        polylines.emplace_back(ring_to_api(tg_poly_hole_at(poly, j)));
      }
      mp.push_back(std::move(polylines));
    }
    return mp;
  };

  auto const add_provider = [&](gbfs::gbfs_provider const* provider) {
    if (!query.withProviders_) {
      return;
    }
    auto form_factors = std::vector<api::RentalFormFactorEnum>{};
    for (auto const& vt : provider->vehicle_types_) {
      auto const ff = gbfs::to_api_form_factor(vt.form_factor_);
      if (utl::find(form_factors, ff) == end(form_factors)) {
        form_factors.push_back(ff);
      }
    }
    res.providers_.emplace_back(api::RentalProvider{
        .id_ = provider->id_,
        .name_ = provider->sys_info_.name_,
        .groupId_ = provider->group_id_,
        .operator_ = provider->sys_info_.operator_,
        .url_ = provider->sys_info_.url_,
        .purchaseUrl_ = provider->sys_info_.purchase_url_,
        .color_ = provider->color_,
        .bbox_ = {provider->bbox_.min_.lng_, provider->bbox_.min_.lat_,
                  provider->bbox_.max_.lng_, provider->bbox_.max_.lat_},
        .vehicleTypes_ = utl::to_vec(
            provider->vehicle_types_,
            [&](gbfs::vehicle_type const& vt) {
              return api::RentalVehicleType{
                  .id_ = vt.id_,
                  .name_ = vt.name_,
                  .formFactor_ = gbfs::to_api_form_factor(vt.form_factor_),
                  .propulsionType_ =
                      gbfs::to_api_propulsion_type(vt.propulsion_type_),
                  .returnConstraint_ =
                      gbfs::to_api_return_constraint(vt.return_constraint_),
                  .returnConstraintGuessed_ = !vt.known_return_constraint_};
            }),
        .formFactors_ = std::move(form_factors),
        .defaultRestrictions_ =
            restrictions_to_api(provider->default_restrictions_),
        .globalGeofencingRules_ =
            utl::to_vec(provider->geofencing_zones_.global_rules_,
                        [&](gbfs::rule const& r) { return rule_to_api(r); })});
  };

  auto const add_provider_group = [&](gbfs::gbfs_group const& group) {
    auto form_factors = std::vector<api::RentalFormFactorEnum>{};
    auto color = group.color_;
    for (auto const& pi : group.providers_) {
      auto const& provider = gbfs->providers_.at(pi);
      if (provider == nullptr) {
        // shouldn't be possible, but just in case...
        std::cerr << "[rental api] warning: provider group " << group.id_
                  << " references missing provider idx " << to_idx(pi) << "\n";
        continue;
      }
      for (auto const& vt : provider->vehicle_types_) {
        auto const ff = gbfs::to_api_form_factor(vt.form_factor_);
        if (utl::find(form_factors, ff) == end(form_factors)) {
          form_factors.push_back(ff);
        }
        if (!color && provider->color_) {
          color = provider->color_;
        }
      }
    }
    auto provider_ids = std::vector<std::string>{};
    if (query.withProviders_) {
      provider_ids.reserve(group.providers_.size());
      for (auto const& pi : group.providers_) {
        auto const& provider = gbfs->providers_.at(pi);
        if (provider == nullptr) {
          // shouldn't be possible, but just in case...
          std::cerr << "[rental api] warning: provider group " << group.id_
                    << " references missing provider idx " << to_idx(pi)
                    << " (providers list)\n";
          continue;
        }
        provider_ids.push_back(provider->id_);
      }
    }

    res.providerGroups_.emplace_back(
        api::RentalProviderGroup{.id_ = group.id_,
                                 .name_ = group.name_,
                                 .color_ = color,
                                 .providers_ = std::move(provider_ids),
                                 .formFactors_ = form_factors});
  };

  if (!filter_bbox && !filter_point && !filter_providers && !filter_groups) {
    for (auto const& provider : gbfs->providers_) {
      if (provider != nullptr) {
        add_provider(provider.get());
      }
    }
    for (auto const& group : gbfs->groups_ | std::views::values) {
      add_provider_group(group);
    }
    return res;
  }

  auto bbox = filter_bbox ? geo::box{min->pos_, max->pos_} : geo::box{};
  auto const in_bbox = [&](geo::latlng const& pos) {
    return filter_bbox ? bbox.contains(pos) : true;
  };

  auto const approx_distance_lng_degrees =
      point_pos ? geo::approx_distance_lng_degrees(*point_pos) : 0.0;
  auto const point_radius_squared =
      point_radius ? (*point_radius) * (*point_radius) : 0.0;
  auto const in_radius = [&](geo::latlng const& pos) {
    return filter_point ? geo::approx_squared_distance(
                              *point_pos, pos, approx_distance_lng_degrees) <=
                              point_radius_squared
                        : true;
  };

  auto const include_bbox = [&](geo::box const& b) {
    if (filter_bbox && !b.overlaps(bbox)) {
      return false;
    }
    if (filter_point) {
      auto const closest =
          geo::latlng{std::clamp(point_pos->lat(), b.min_.lat(), b.max_.lat()),
                      std::clamp(point_pos->lng(), b.min_.lng(), b.max_.lng())};
      return geo::approx_squared_distance(*point_pos, closest,
                                          approx_distance_lng_degrees) <=
             point_radius_squared;
    }
    return true;
  };

  auto providers = hash_set<gbfs::gbfs_provider const*>{};
  auto provider_groups = hash_set<std::string>{};

  auto check_provider = [&](gbfs_provider_idx_t const pi) {
    auto const& provider = gbfs->providers_.at(pi);
    if (provider == nullptr || providers.contains(provider.get())) {
      return;
    }
    if ((filter_providers || filter_groups) &&
        !((filter_providers && utl::find(*query.providers_, provider->id_) !=
                                   end(*query.providers_)) ||
          (filter_groups &&
           utl::find(*query.providerGroups_, provider->group_id_) !=
               end(*query.providerGroups_)))) {
      return;
    }
    providers.insert(provider.get());
    provider_groups.insert(provider->group_id_);
  };

  if (filter_point) {
    gbfs->provider_rtree_.in_radius(
        *point_pos, *point_radius,
        [&](gbfs_provider_idx_t const pi) { check_provider(pi); });
    gbfs->provider_zone_rtree_.in_radius(
        *point_pos, *point_radius,
        [&](gbfs_provider_idx_t const pi) { check_provider(pi); });
  } else if (filter_bbox) {
    gbfs->provider_rtree_.find(
        bbox, [&](gbfs_provider_idx_t const pi) { check_provider(pi); });
    gbfs->provider_zone_rtree_.find(
        bbox, [&](gbfs_provider_idx_t const pi) { check_provider(pi); });
  } else if (filter_providers || filter_groups) {
    if (filter_providers) {
      for (auto const& id : *query.providers_) {
        if (auto const it = gbfs->provider_by_id_.find(id);
            it != end(gbfs->provider_by_id_)) {
          auto const& provider = gbfs->providers_.at(it->second);
          if (provider != nullptr) {
            providers.insert(provider.get());
            provider_groups.insert(provider->group_id_);
          }
        }
      }
    }
    if (filter_groups) {
      for (auto const& id : *query.providerGroups_) {
        if (auto const it = gbfs->groups_.find(id); it != end(gbfs->groups_)) {
          provider_groups.insert(it->second.id_);
          for (auto const pi : it->second.providers_) {
            auto const& provider = gbfs->providers_.at(pi);
            if (provider == nullptr) {
              // shouldn't be possible, but just in case...
              std::cerr << "[rental api] warning: provider group "
                        << it->second.id_ << " references missing provider idx "
                        << to_idx(pi) << "\n";
              continue;
            }
            providers.insert(provider.get());
          }
        }
      }
    }
  }

  for (auto const* provider : providers) {
    add_provider(provider);

    if (query.withStations_) {
      for (auto const& st : provider->stations_ | std::views::values) {
        auto const sbb = st.info_.bounding_box();
        if (!include_bbox(sbb)) {
          continue;
        }
        auto form_factor_counts =
            std::array<std::uint64_t,
                       std::to_underlying(api::RentalFormFactorEnum::OTHER) +
                           1>{};
        auto types_available = std::map<std::string, std::uint64_t>{};
        auto docks_available = std::map<std::string, std::uint64_t>{};
        auto form_factors = std::set<api::RentalFormFactorEnum>{};

        for (auto const& [vti, count] : st.status_.vehicle_types_available_) {
          if (vti == gbfs::vehicle_type_idx_t::invalid() ||
              cista::to_idx(vti) >= provider->vehicle_types_.size()) {
            continue;
          }
          auto const& vt = provider->vehicle_types_.at(vti);
          auto const api_ff = gbfs::to_api_form_factor(vt.form_factor_);
          form_factor_counts[static_cast<std::size_t>(
              std::to_underlying(api_ff))] += count;
          types_available[vt.id_] = count;
          form_factors.insert(api_ff);
        }
        for (auto const& [vti, count] : st.status_.vehicle_docks_available_) {
          if (vti == gbfs::vehicle_type_idx_t::invalid() ||
              cista::to_idx(vti) >= provider->vehicle_types_.size()) {
            continue;
          }
          auto const& vt = provider->vehicle_types_.at(vti);
          auto const api_ff = gbfs::to_api_form_factor(vt.form_factor_);
          form_factor_counts[static_cast<std::size_t>(
              std::to_underlying(api_ff))] += count;
          docks_available[vt.id_] = count;
          form_factors.insert(api_ff);
        }

        if (form_factors.empty()) {
          for (auto const& vt : provider->vehicle_types_) {
            form_factors.insert(gbfs::to_api_form_factor(vt.form_factor_));
          }
        }

        auto sorted_form_factors = utl::to_vec(form_factors);
        utl::sort(sorted_form_factors, [&](auto const a, auto const b) {
          return form_factor_counts[static_cast<std::size_t>(
                     std::to_underlying(a))] >
                 form_factor_counts[static_cast<std::size_t>(
                     std::to_underlying(b))];
        });

        res.stations_.emplace_back(api::RentalStation{
            .id_ = st.info_.id_,
            .providerId_ = provider->id_,
            .providerGroupId_ = provider->group_id_,
            .name_ = st.info_.name_,
            .lat_ = st.info_.pos_.lat_,
            .lon_ = st.info_.pos_.lng_,
            .address_ = st.info_.address_,
            .crossStreet_ = st.info_.cross_street_,
            .rentalUriAndroid_ = st.info_.rental_uris_.android_,
            .rentalUriIOS_ = st.info_.rental_uris_.ios_,
            .rentalUriWeb_ = st.info_.rental_uris_.web_,
            .isRenting_ = st.status_.is_renting_,
            .isReturning_ = st.status_.is_returning_,
            .numVehiclesAvailable_ = st.status_.num_vehicles_available_,
            .formFactors_ = sorted_form_factors,
            .vehicleTypesAvailable_ = std::move(types_available),
            .vehicleDocksAvailable_ = std::move(docks_available),
            .stationArea_ = st.info_.station_area_ != nullptr
                                ? std::optional{multipoly_to_api(
                                      st.info_.station_area_.get())}
                                : std::nullopt,
            .bbox_ = {sbb.min_.lng_, sbb.min_.lat_, sbb.max_.lng_,
                      sbb.max_.lat_}});
      }
    }

    if (query.withVehicles_) {
      auto const add_vehicle = [&](gbfs::vehicle_status const& vs,
                                   gbfs::vehicle_type const& vt) {
        res.vehicles_.emplace_back(api::RentalVehicle{
            .id_ = vs.id_,
            .providerId_ = provider->id_,
            .providerGroupId_ = provider->group_id_,
            .typeId_ = vt.id_,
            .lat_ = vs.pos_.lat_,
            .lon_ = vs.pos_.lng_,
            .formFactor_ = gbfs::to_api_form_factor(vt.form_factor_),
            .propulsionType_ =
                gbfs::to_api_propulsion_type(vt.propulsion_type_),
            .returnConstraint_ =
                gbfs::to_api_return_constraint(vt.return_constraint_),
            .stationId_ = vs.station_id_,
            .homeStationId_ = vs.home_station_id_,
            .isReserved_ = vs.is_reserved_,
            .isDisabled_ = vs.is_disabled_,
            .rentalUriAndroid_ = vs.rental_uris_.android_,
            .rentalUriIOS_ = vs.rental_uris_.ios_,
            .rentalUriWeb_ = vs.rental_uris_.web_,
        });
      };
      auto const fallback_vt =
          gbfs::vehicle_type{.form_factor_ = gbfs::vehicle_form_factor::kOther};
      for (auto const& vs : provider->vehicle_status_) {
        if (in_bbox(vs.pos_) && in_radius(vs.pos_)) {
          if (vs.vehicle_type_idx_ != gbfs::vehicle_type_idx_t::invalid() &&
              cista::to_idx(vs.vehicle_type_idx_) <
                  provider->vehicle_types_.size()) {
            auto const& vt = provider->vehicle_types_.at(vs.vehicle_type_idx_);
            add_vehicle(vs, vt);
          } else {
            add_vehicle(vs, fallback_vt);
          }
        }
      }
    }

    if (query.withZones_) {
      auto const n_zones =
          static_cast<std::int64_t>(provider->geofencing_zones_.zones_.size());
      for (auto const [order, zone] :
           utl::enumerate(provider->geofencing_zones_.zones_)) {
        auto const zbb = zone.bounding_box();
        if (!include_bbox(zbb)) {
          continue;
        }
        res.zones_.emplace_back(api::RentalZone{
            .providerId_ = provider->id_,
            .providerGroupId_ = provider->group_id_,
            .name_ = zone.name_,
            .z_ = n_zones - static_cast<std::int64_t>(order),
            .bbox_ = {zbb.min_.lng_, zbb.min_.lat_, zbb.max_.lng_,
                      zbb.max_.lat_},
            .area_ = multipoly_to_api(zone.geom_.get()),
            .rules_ = utl::to_vec(
                zone.rules_,
                [&](gbfs::rule const& r) { return rule_to_api(r); }),
        });
      }
    }
  }

  for (auto const& group_id : provider_groups) {
    add_provider_group(gbfs->groups_.at(group_id));
  }

  return res;
}

}  // namespace motis::ep
