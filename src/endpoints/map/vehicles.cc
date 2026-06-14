#include "motis/endpoints/map/vehicles.h"

#include <chrono>

#include "date/date.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "net/bad_request_exception.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/types.h"

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/parse_location.h"
#include "motis/rt/vehicle_position.h"
#include "motis/tag_lookup.h"

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

std::optional<std::string> resolve_headsign(
    tag_lookup const& tags,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    vehicle_positions::vehicle_position const& vehicle,
    n::lang_t const& lang) {
  auto const src = tags.get_src(feed_tag(vehicle.feed_id_));
  if (src == n::source_idx_t::invalid()) {
    return std::nullopt;
  }

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
    auto const fr = n::rt::frun{tt, rtt, run};
    if (fr.size() == 0U) {
      return std::nullopt;
    }
    return std::string{fr[0].direction(lang, n::event_type::kDep)};
  } catch (...) {
    return std::nullopt;
  }
}

api::VehiclePosition to_api(
    vehicle_positions::vehicle_position const& vehicle,
    std::optional<std::string> headsign) {
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
              .startDate_ = vehicle.trip_.start_date_,
              .startTime_ = vehicle.trip_.start_time_,
              .routeId_ = vehicle.trip_.route_id_,
              .headsign_ = std::move(headsign),
              .directionId_ = to_i64(vehicle.trip_.direction_id_),
              .scheduleRelationship_ = vehicle.trip_.schedule_relationship_},
      .reportedPosition_ =
          api::ReportedVehiclePosition{
              .lat_ = vehicle.reported_position_.pos_.lat_,
              .lon_ = vehicle.reported_position_.pos_.lng_,
              .bearing_ = vehicle.reported_position_.bearing_,
              .speedMps_ = vehicle.reported_position_.speed_mps_},
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
    auto const headsign =
        tags_ != nullptr && tt_ != nullptr
            ? resolve_headsign(*tags_, *tt_, rtt, vehicle, query.language_)
            : std::nullopt;
    res.vehicles_.emplace_back(to_api(vehicle, headsign));
  }
  return res;
}

}  // namespace motis::ep
