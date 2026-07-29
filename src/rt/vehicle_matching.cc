#include "motis/rt/vehicle_matching.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "date/date.h"

#include "cista/hash.h"

#include "fmt/core.h"

#include "geo/polyline_format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/loader/gtfs/route.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/types.h"

#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::vehicle_matching {

namespace {

std::string_view feed_tag(std::string_view const feed_id) {
  auto const pos = feed_id.find(':');
  return pos == std::string_view::npos ? feed_id : feed_id.substr(0U, pos);
}

std::optional<std::int64_t> to_i64(std::optional<std::uint32_t> const x) {
  return x.has_value() ? std::optional<std::int64_t>{*x} : std::nullopt;
}

transit_realtime::TripDescriptor to_trip_descriptor(
    vehicle_positions::vehicle_position const& vehicle) {
  auto td = transit_realtime::TripDescriptor{};
  if (vehicle.trip_.trip_id_.has_value()) {
    td.set_trip_id(*vehicle.trip_.trip_id_);
  }
  if (vehicle.trip_.start_date_.has_value()) {
    td.set_start_date(*vehicle.trip_.start_date_);
  }
  if (vehicle.trip_.start_time_.has_value()) {
    td.set_start_time(*vehicle.trip_.start_time_);
  }
  if (vehicle.trip_.route_id_.has_value()) {
    td.set_route_id(*vehicle.trip_.route_id_);
  }
  if (vehicle.trip_.direction_id_.has_value()) {
    td.set_direction_id(*vehicle.trip_.direction_id_);
  }
  return td;
}

std::optional<n::trip_idx_t> find_static_trip(
    n::timetable const& tt,
    n::source_idx_t const src,
    std::string_view const trip_id) {
  auto const it = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](n::pair<n::trip_id_idx_t, n::trip_idx_t> const& entry,
          std::string_view const id) {
        return std::tuple{tt.trip_id_src_[entry.first],
                          tt.trip_id_strings_[entry.first].view()} <
               std::tuple{src, id};
      });
  if (it != end(tt.trip_id_to_idx_) && tt.trip_id_src_[it->first] == src &&
      tt.trip_id_strings_[it->first].view() == trip_id) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<n::rt::frun> static_trip_run(n::timetable const& tt,
                                           n::trip_idx_t const trip) {
  if (tt.trip_transport_ranges_[trip].empty()) {
    return std::nullopt;
  }
  auto const [transport, stop_range] = tt.trip_transport_ranges_[trip].front();
  return n::rt::frun{
      tt, nullptr,
      n::rt::run{.t_ = n::transport{transport, n::day_idx_t{0U}},
                 .stop_range_ = stop_range,
                 .rt_ = n::rt_transport_idx_t::invalid()}};
}

api::TransitVehicleRouteInfo to_route_info(n::rt::run_stop const& s,
                                           n::lang_t const& lang) {
  auto const color = s.get_route_color(n::event_type::kDep);
  return api::TransitVehicleRouteInfo{
      .id_ = std::string{s.get_route_id(n::event_type::kDep)},
      .shortName_ =
          std::string{s.route_short_name(n::event_type::kDep, lang)},
      .longName_ = std::string{s.route_long_name(n::event_type::kDep, lang)},
      .color_ = to_str(color.color_),
      .textColor_ = to_str(color.text_color_)};
}

std::optional<n::route_id_idx_t> find_route_id(
    n::timetable::route_ids const& routes,
    std::string_view const realtime_route_id) {
  if (auto const exact = routes.ids_.find(realtime_route_id);
      exact.has_value()) {
    return exact;
  }

  auto best = std::optional<n::route_id_idx_t>{};
  auto best_len = std::size_t{0U};
  for (auto i = std::uint32_t{0U}; i < routes.ids_.size(); ++i) {
    auto const route_id = n::route_id_idx_t{i};
    auto const static_route_id = routes.ids_.get(route_id);
    if (static_route_id.size() <= best_len) {
      continue;
    }
    if (realtime_route_id == static_route_id ||
        (static_route_id.size() < realtime_route_id.size() &&
         realtime_route_id.ends_with(static_route_id))) {
      best = route_id;
      best_len = static_route_id.size();
    }
  }
  return best;
}

api::VehicleShapeSourceEnum shape_source(n::rt::frun const& fr,
                                         n::shapes_storage const* shapes) {
  if (!fr.is_scheduled() || shapes == nullptr) {
    return api::VehicleShapeSourceEnum::NONE;
  }
  auto const shape_idx = shapes->get_shape_idx(fr.trip_idx());
  if (shape_idx == n::scoped_shape_idx_t::invalid()) {
    return api::VehicleShapeSourceEnum::NONE;
  }
  switch (n::get_shape_source(shape_idx)) {
    case n::shape_source::kNone: return api::VehicleShapeSourceEnum::NONE;
    case n::shape_source::kTimetable:
      return api::VehicleShapeSourceEnum::TIMETABLE;
    case n::shape_source::kRouted: return api::VehicleShapeSourceEnum::ROUTED;
  }
  std::unreachable();
}

std::optional<api::EncodedPolyline> encode_shape(
    n::rt::frun const& fr,
    n::shapes_storage const* shapes) {
  if (fr.size() < 2U) {
    return std::nullopt;
  }

  auto enc = geo::polyline_encoder<6>{};
  auto n_points = std::int64_t{0};
  fr.for_each_shape_point(
      shapes,
      n::interval{n::stop_idx_t{0U}, static_cast<n::stop_idx_t>(fr.size())},
      [&](geo::latlng const& p) {
        enc.push_nonzero_diff(p, 2);
        ++n_points;
      });
  if (n_points == 0) {
    return std::nullopt;
  }
  return api::EncodedPolyline{
      .points_ = std::move(enc.buf_), .precision_ = 6, .length_ = n_points};
}

void apply_trip_run_details(vehicle_details& details,
                            tag_lookup const& tags,
                            n::timetable const& tt,
                            n::rt::frun const& fr,
                            n::shapes_storage const* shapes,
                            n::lang_t const& lang) {
  auto const first = fr[0];
  details.scheduled_trip_id_ = tags.id(tt, first, n::event_type::kDep);
  details.headsign_ = std::string{first.direction(lang, n::event_type::kDep)};
  details.route_ = to_route_info(first, lang);
  details.mode_ = to_mode(first.get_clasz(n::event_type::kDep), 5);
  details.shape_ = encode_shape(fr, shapes);
  if (details.shape_.has_value()) {
    auto const hash = cista::hash_combine(
        cista::hash(details.shape_->points_), details.shape_->precision_,
        details.shape_->length_);
    details.shape_id_ = fmt::format("vehicle-shape-{:016x}", hash);
    details.shape_source_ = shape_source(fr, shapes);
  }
}

bool stop_matches(n::rt::frun const& target,
                  std::optional<std::string> const& stop_id) {
  if (!stop_id.has_value()) {
    return false;
  }
  for (auto i = n::stop_idx_t{0U}; i != target.size(); ++i) {
    auto const stop = target[i];
    auto const id = stop.get_location_id();
    if (id == *stop_id ||
        (id.size() > stop_id->size() && id.ends_with(*stop_id) &&
         id[id.size() - stop_id->size() - 1U] == '_')) {
      return true;
    }
  }
  return false;
}

struct ranked_vehicle {
  vehicle_positions::vehicle_position const* vehicle_;
  vehicle_details details_;
  bool exact_trip_match_;
  std::uint8_t vehicle_id_consistency_;
  std::uint8_t route_direction_stop_consistency_;
  std::int64_t freshness_;
};

bool better(ranked_vehicle const& a, ranked_vehicle const& b) {
  auto const a_rank =
      std::tuple{a.exact_trip_match_, a.vehicle_id_consistency_,
                 a.route_direction_stop_consistency_, a.freshness_};
  auto const b_rank =
      std::tuple{b.exact_trip_match_, b.vehicle_id_consistency_,
                 b.route_direction_stop_consistency_, b.freshness_};
  if (a_rank != b_rank) {
    return a_rank > b_rank;
  }
  return std::tie(a.vehicle_->feed_id_, a.vehicle_->entity_id_) <
         std::tie(b.vehicle_->feed_id_, b.vehicle_->entity_id_);
}

}  // namespace

std::optional<n::rt::frun> resolve_run(
    tag_lookup const& tags,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    vehicle_positions::vehicle_position const& vehicle) {
  auto const src = tags.get_src(feed_tag(vehicle.feed_id_));
  if (src == n::source_idx_t::invalid()) {
    return std::nullopt;
  }

  auto const td = to_trip_descriptor(vehicle);
  if (!td.has_trip_id() &&
      !(td.has_route_id() && td.has_direction_id() && td.has_start_date() &&
        td.has_start_time())) {
    return std::nullopt;
  }

  try {
    auto const today = std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
    auto const [run, _] = n::rt::gtfsrt_resolve_run(today, tt, rtt, src, td);
    if (!run.valid()) {
      return std::nullopt;
    }
    auto fr = n::rt::frun{tt, rtt, run};
    if (fr.size() == 0U) {
      return std::nullopt;
    }
    return fr;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<n::rt::frun> resolve_static_trip_run_by_id(
    tag_lookup const& tags,
    n::timetable const& tt,
    vehicle_positions::vehicle_position const& vehicle) {
  if (!vehicle.trip_.trip_id_.has_value()) {
    return std::nullopt;
  }
  auto const src = tags.get_src(feed_tag(vehicle.feed_id_));
  if (src == n::source_idx_t::invalid()) {
    return std::nullopt;
  }
  auto const trip = find_static_trip(tt, src, *vehicle.trip_.trip_id_);
  if (!trip.has_value()) {
    return std::nullopt;
  }

  return static_trip_run(tt, *trip);
}

std::optional<api::TransitVehicleRouteInfo> resolve_route_info_by_id(
    tag_lookup const& tags,
    n::timetable const& tt,
    vehicle_positions::vehicle_position const& vehicle,
    n::lang_t const& lang) {
  if (!vehicle.trip_.route_id_.has_value()) {
    return std::nullopt;
  }
  auto const src = tags.get_src(feed_tag(vehicle.feed_id_));
  if (src == n::source_idx_t::invalid() || src >= tt.route_ids_.size()) {
    return std::nullopt;
  }
  auto const& routes = tt.route_ids_[src];
  auto const route_id = find_route_id(routes, *vehicle.trip_.route_id_);
  if (!route_id.has_value()) {
    return std::nullopt;
  }

  auto const color = routes.route_id_colors_[*route_id];
  return api::TransitVehicleRouteInfo{
      .id_ = *vehicle.trip_.route_id_,
      .shortName_ = std::string{
          tt.translate(lang, routes.route_id_short_names_[*route_id])},
      .longName_ = std::string{
          tt.translate(lang, routes.route_id_long_names_[*route_id])},
      .color_ = to_str(color.color_),
      .textColor_ = to_str(color.text_color_)};
}

std::optional<api::ModeEnum> resolve_mode_by_route_id(
    tag_lookup const& tags,
    n::timetable const& tt,
    vehicle_positions::vehicle_position const& vehicle) {
  if (!vehicle.trip_.route_id_.has_value()) {
    return std::nullopt;
  }
  auto const src = tags.get_src(feed_tag(vehicle.feed_id_));
  if (src == n::source_idx_t::invalid() || src >= tt.route_ids_.size()) {
    return std::nullopt;
  }
  auto const& routes = tt.route_ids_[src];
  auto const route_id = find_route_id(routes, *vehicle.trip_.route_id_);
  if (!route_id.has_value()) {
    return std::nullopt;
  }
  return to_mode(n::loader::gtfs::to_clasz(routes.route_id_type_[*route_id]),
                 5);
}

vehicle_details resolve_details(
    tag_lookup const* tags,
    n::timetable const* tt,
    n::rt_timetable const* rtt,
    n::shapes_storage const* shapes,
    vehicle_positions::vehicle_position const& v,
    n::lang_t const& lang) {
  auto details = vehicle_details{};
  if (tags == nullptr || tt == nullptr) {
    return details;
  }

  if (auto fr = resolve_run(*tags, *tt, rtt, v); fr.has_value()) {
    apply_trip_run_details(details, *tags, *tt, *fr, shapes, lang);
    details.match_state_ = api::VehicleMatchStateEnum::MATCHED_TRIP;
  }
  if (!details.route_.has_value()) {
    if (auto fr = resolve_static_trip_run_by_id(*tags, *tt, v);
        fr.has_value()) {
      apply_trip_run_details(details, *tags, *tt, *fr, shapes, lang);
      details.match_state_ =
          api::VehicleMatchStateEnum::MATCHED_ROUTE_ONLY;
    }
  }
  if (!details.route_.has_value()) {
    details.route_ = resolve_route_info_by_id(*tags, *tt, v, lang);
    if (details.route_.has_value()) {
      details.match_state_ =
          api::VehicleMatchStateEnum::MATCHED_ROUTE_ONLY;
    }
  }
  if (!details.mode_.has_value()) {
    details.mode_ = resolve_mode_by_route_id(*tags, *tt, v);
  }
  return details;
}

api::VehiclePosition to_api(
    vehicle_positions::vehicle_position const& vehicle,
    vehicle_details details,
    bool const include_shapes) {
  return api::VehiclePosition{
      .feedId_ = vehicle.feed_id_,
      .entityId_ = vehicle.entity_id_,
      .vehicle_ =
          api::TransitVehicleDescriptor{
              .id_ = vehicle.vehicle_.id_,
              .label_ = vehicle.vehicle_.label_,
              .licensePlate_ = vehicle.vehicle_.license_plate_,
              .wheelchairAccessible_ =
                  vehicle.vehicle_.wheelchair_accessible_},
      .trip_ =
          api::TransitVehicleTripDescriptor{
              .tripId_ = vehicle.trip_.trip_id_,
              .scheduledTripId_ = std::move(details.scheduled_trip_id_),
              .startDate_ = vehicle.trip_.start_date_,
              .startTime_ = vehicle.trip_.start_time_,
              .routeId_ = vehicle.trip_.route_id_,
              .headsign_ = std::move(details.headsign_),
              .directionId_ = to_i64(vehicle.trip_.direction_id_),
              .scheduleRelationship_ = vehicle.trip_.schedule_relationship_},
      .matchState_ = details.match_state_,
      .route_ = std::move(details.route_),
      .reportedPosition_ =
          api::ReportedVehiclePosition{
              .lat_ = vehicle.reported_position_.pos_.lat_,
              .lon_ = vehicle.reported_position_.pos_.lng_,
              .bearing_ = vehicle.reported_position_.bearing_,
              .speedMps_ = vehicle.reported_position_.speed_mps_},
      .mode_ = std::move(details.mode_),
      .shape_ = include_shapes ? std::move(details.shape_) : std::nullopt,
      .shapeId_ = std::move(details.shape_id_),
      .shapeSource_ = std::move(details.shape_source_),
      .currentStopSequence_ = to_i64(vehicle.current_stop_sequence_),
      .stopId_ = vehicle.stop_id_,
      .currentStatus_ = vehicle.current_status_,
      .occupancyStatus_ = vehicle.occupancy_status_,
      .reportedTime_ = vehicle.reported_time_,
      .ingestedTime_ = vehicle.ingested_time_};
}

std::optional<api::VehiclePosition> primary_vehicle(
    tag_lookup const& tags,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::shapes_storage const* shapes,
    vehicle_positions::vehicle_position_store const& store,
    n::rt::frun const& target,
    n::lang_t const& lang) {
  auto best = std::optional<ranked_vehicle>{};
  auto const first = target[0];
  auto const target_route = first.get_route_id(n::event_type::kDep);
  auto const target_direction = first.get_direction_id(n::event_type::kDep);

  for (auto const& vehicle : store.all()) {
    auto exact_trip_match = false;
    auto matches_target = false;
    if (auto fr = resolve_run(tags, tt, rtt, vehicle); fr.has_value()) {
      exact_trip_match =
          fr->trip_idx() == target.trip_idx() && fr->t_ == target.t_;
      matches_target = exact_trip_match;
    }
    if (!matches_target) {
      if (auto fr = resolve_static_trip_run_by_id(tags, tt, vehicle);
          fr.has_value()) {
        matches_target = fr->trip_idx() == target.trip_idx();
      }
    }
    if (!matches_target) {
      continue;
    }

    auto details = resolve_details(&tags, &tt, rtt, shapes, vehicle, lang);
    auto consistency = std::uint8_t{0U};
    consistency += vehicle.trip_.route_id_ == target_route;
    consistency += vehicle.trip_.direction_id_.has_value() &&
                   *vehicle.trip_.direction_id_ ==
                       static_cast<std::uint32_t>(target_direction.v_);
    consistency += stop_matches(target, vehicle.stop_id_);
    auto candidate = ranked_vehicle{
        .vehicle_ = &vehicle,
        .details_ = std::move(details),
        .exact_trip_match_ = exact_trip_match,
        .vehicle_id_consistency_ = static_cast<std::uint8_t>(
            !vehicle.vehicle_.id_.has_value()
                ? 0U
                : (*vehicle.vehicle_.id_ == vehicle.entity_id_ ? 2U : 1U)),
        .route_direction_stop_consistency_ = consistency,
        .freshness_ = vehicle.reported_time_.value_or(
            std::numeric_limits<std::int64_t>::min())};
    if (!best.has_value() || better(candidate, *best)) {
      best = std::move(candidate);
    }
  }

  if (!best.has_value()) {
    return std::nullopt;
  }
  return to_api(*best->vehicle_, std::move(best->details_), false);
}

}  // namespace motis::vehicle_matching
