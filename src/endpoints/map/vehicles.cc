#include "motis/endpoints/map/vehicles.h"

#include <chrono>

#include "date/date.h"

#include "geo/polyline_format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "net/bad_request_exception.h"

#include "nigiri/loader/gtfs/route.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/types.h"

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/parse_location.h"
#include "motis/rt/vehicle_position.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

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

std::optional<n::trip_idx_t> find_static_trip(
    n::timetable const& tt,
    n::source_idx_t const src,
    std::string_view const trip_id) {
  for (auto const& [id_idx, trip_idx] : tt.trip_id_to_idx_) {
    if (tt.trip_id_src_[id_idx] == src &&
        tt.trip_id_strings_[id_idx].view() == trip_id) {
      return trip_idx;
    }
  }
  return std::nullopt;
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
  if (!trip.has_value() || tt.trip_transport_ranges_[*trip].empty()) {
    return std::nullopt;
  }

  auto const [transport, stop_range] = tt.trip_transport_ranges_[*trip].front();
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

bool route_id_matches(std::string_view const realtime_route_id,
                      std::string_view const static_route_id) {
  return realtime_route_id == static_route_id ||
         (static_route_id.size() < realtime_route_id.size() &&
          realtime_route_id.ends_with(static_route_id));
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

std::optional<n::rt::frun> resolve_route_shape_run_by_id(
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

  auto const trip = routes.route_id_trips_[*route_id].front();
  if (tt.trip_transport_ranges_[trip].empty()) {
    return std::nullopt;
  }

  auto const transport = tt.trip_transport_ranges_[trip].front().first;
  auto const route = tt.transport_route_[transport];
  return n::rt::frun{
      tt, nullptr,
      n::rt::run{
          .t_ = n::transport{transport, n::day_idx_t{0U}},
          .stop_range_ =
              n::interval{n::stop_idx_t{0U},
                          static_cast<n::stop_idx_t>(
                              tt.route_location_seq_[route].size())},
          .rt_ = n::rt_transport_idx_t::invalid()}};
}

std::optional<n::route_idx_t> resolve_static_route_by_id(
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
  if (!route_id.has_value() || routes.route_id_trips_[*route_id].empty()) {
    return std::nullopt;
  }

  if (!routes.route_id_trips_[*route_id].empty()) {
    auto const trip = routes.route_id_trips_[*route_id].front();
    if (!tt.trip_transport_ranges_[trip].empty()) {
      return tt.transport_route_[tt.trip_transport_ranges_[trip].front().first];
    }
  }

  for (auto i = std::uint32_t{0U}; i < tt.n_routes(); ++i) {
    auto const route = n::route_idx_t{i};
    if (tt.route_transport_ranges_[route].empty()) {
      continue;
    }
    auto const transport = tt.route_transport_ranges_[route].from_;
    auto const fr = n::rt::frun{
        tt, nullptr,
        n::rt::run{
            .t_ = n::transport{transport, n::day_idx_t{0U}},
            .stop_range_ = n::interval{n::stop_idx_t{0U}, n::stop_idx_t{2U}},
            .rt_ = n::rt_transport_idx_t::invalid()}};
    if (fr.size() >= 1U &&
        route_id_matches(*vehicle.trip_.route_id_,
                         fr[0].get_route_id(n::event_type::kDep))) {
      return route;
    }
  }

  return std::nullopt;
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

std::optional<api::EncodedPolyline> encode_route_stop_shape(
    n::timetable const& tt,
    n::route_idx_t const route) {
  auto const stops = tt.route_location_seq_[route];
  if (stops.size() < 2U) {
    return std::nullopt;
  }

  auto enc = geo::polyline_encoder<6>{};
  auto n_points = std::int64_t{0};
  for (auto const stop : stops) {
    enc.push_nonzero_diff(
        tt.locations_.coordinates_[n::stop{stop}.location_idx()], 2);
    ++n_points;
  }
  return api::EncodedPolyline{
      .points_ = std::move(enc.buf_), .precision_ = 6, .length_ = n_points};
}

struct vehicle_details {
  std::optional<std::string> scheduled_trip_id_;
  std::optional<std::string> headsign_;
  std::optional<api::TransitVehicleRouteInfo> route_;
  std::optional<api::ModeEnum> mode_;
  std::optional<api::EncodedPolyline> shape_;
  std::optional<api::VehicleShapeSourceEnum> shape_source_;
};

vehicle_details resolve_details(tag_lookup const* tags,
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
    auto const first = (*fr)[0];
    details.scheduled_trip_id_ =
        tags->id(*tt, first, n::event_type::kDep);
    details.headsign_ =
        std::string{first.direction(lang, n::event_type::kDep)};
    details.route_ = to_route_info(first, lang);
    details.mode_ = to_mode(first.get_clasz(n::event_type::kDep), 5);
    details.shape_ = encode_shape(*fr, shapes);
    if (details.shape_.has_value()) {
      details.shape_source_ = shape_source(*fr, shapes);
    }
  }
  if (!details.route_.has_value()) {
    if (auto fr = resolve_static_trip_run_by_id(*tags, *tt, v);
        fr.has_value()) {
      auto const first = (*fr)[0];
      details.scheduled_trip_id_ =
          tags->id(*tt, first, n::event_type::kDep);
      details.headsign_ =
          std::string{first.direction(lang, n::event_type::kDep)};
      details.route_ = to_route_info(first, lang);
      details.mode_ = to_mode(first.get_clasz(n::event_type::kDep), 5);
      details.shape_ = encode_shape(*fr, shapes);
      if (details.shape_.has_value()) {
        details.shape_source_ = shape_source(*fr, shapes);
      }
    }
  }

  if (!details.route_.has_value()) {
    details.route_ = resolve_route_info_by_id(*tags, *tt, v, lang);
  }
  if (!details.mode_.has_value()) {
    details.mode_ = resolve_mode_by_route_id(*tags, *tt, v);
  }
  if (!details.shape_.has_value()) {
    if (auto fr = resolve_route_shape_run_by_id(*tags, *tt, v);
        fr.has_value()) {
      details.shape_ = encode_shape(*fr, shapes);
      if (details.shape_.has_value()) {
        details.shape_source_ = shape_source(*fr, shapes);
      }
    }
  }
  if (!details.shape_.has_value()) {
    if (auto route = resolve_static_route_by_id(*tags, *tt, v);
        route.has_value()) {
      details.shape_ = encode_route_stop_shape(*tt, *route);
      if (details.shape_.has_value()) {
        details.shape_source_ = api::VehicleShapeSourceEnum::NONE;
      }
    }
  }
  return details;
}

api::VehiclePosition to_api(
    vehicle_positions::vehicle_position const& vehicle,
    vehicle_details details) {
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
      .route_ = std::move(details.route_),
      .reportedPosition_ =
          api::ReportedVehiclePosition{
              .lat_ = vehicle.reported_position_.pos_.lat_,
              .lon_ = vehicle.reported_position_.pos_.lng_,
              .bearing_ = vehicle.reported_position_.bearing_,
              .speedMps_ = vehicle.reported_position_.speed_mps_},
      .mode_ = std::move(details.mode_),
      .shape_ = std::move(details.shape_),
      .shapeSource_ = std::move(details.shape_source_),
      .currentStopSequence_ = to_i64(vehicle.current_stop_sequence_),
      .stopId_ = vehicle.stop_id_,
      .currentStatus_ = vehicle.current_status_,
      .occupancyStatus_ = vehicle.occupancy_status_,
      .reportedTime_ = vehicle.reported_time_,
      .ingestedTime_ = vehicle.ingested_time_};
}

}  // namespace

api::VehiclePositionsResponse vehicles::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::vehicles_params{url.params()};
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  utl::verify<net::bad_request_exception>(
      min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify<net::bad_request_exception>(
      max.has_value(), "max not a coordinate: {}", query.max_);

  auto const rt = std::atomic_load(&rt_);
  auto const rtt = rt->rtt_.get();
  auto res = api::VehiclePositionsResponse{};
  if (rt->vehicle_positions_ == nullptr) {
    return res;
  }

  auto const snapshot = rt->vehicle_positions_->snapshot(
      vehicle_positions::vehicle_viewport{.min_ = min->pos_, .max_ = max->pos_});
  res.vehicles_.reserve(snapshot.size());
  for (auto const& vehicle : snapshot) {
    res.vehicles_.emplace_back(
        to_api(vehicle, resolve_details(tags_, tt_, rtt, shapes_, vehicle,
                                        query.language_)));
  }
  return res;
}

}  // namespace motis::ep
