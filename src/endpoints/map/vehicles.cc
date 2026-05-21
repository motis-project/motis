#include "motis/endpoints/map/vehicles.h"

#include "net/bad_request_exception.h"

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/parse_location.h"
#include "motis/rt/vehicle_position.h"

namespace motis::ep {

namespace {

api::VehiclePosition to_api(
    vehicle_positions::vehicle_position const& vehicle) {
  return api::VehiclePosition{
      .feedId_ = vehicle.feed_id_,
      .entityId_ = vehicle.entity_id_,
      .vehicle_ =
          api::TransitVehicleDescriptor{
              .id_ = vehicle.vehicle_.id_,
              .label_ = vehicle.vehicle_.label_,
              .licensePlate_ = vehicle.vehicle_.license_plate_},
      .trip_ =
          api::TransitVehicleTripDescriptor{
              .tripId_ = vehicle.trip_.trip_id_,
              .startDate_ = vehicle.trip_.start_date_,
              .startTime_ = vehicle.trip_.start_time_,
              .routeId_ = vehicle.trip_.route_id_,
              .directionId_ =
                  vehicle.trip_.direction_id_.transform([](auto const x) {
                    return static_cast<std::int64_t>(x);
                  })},
      .reportedPosition_ =
          api::ReportedVehiclePosition{
              .lat_ = vehicle.reported_position_.pos_.lat_,
              .lon_ = vehicle.reported_position_.pos_.lng_,
              .bearing_ = vehicle.reported_position_.bearing_,
              .speedMps_ = vehicle.reported_position_.speed_mps_},
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
  auto res = api::VehiclePositionsResponse{};
  if (rt->vehicle_positions_ == nullptr) {
    return res;
  }

  auto const snapshot = rt->vehicle_positions_->snapshot(
      vehicle_positions::vehicle_viewport{.min_ = min->pos_, .max_ = max->pos_});
  res.vehicles_.reserve(snapshot.size());
  for (auto const& vehicle : snapshot) {
    res.vehicles_.emplace_back(to_api(vehicle));
  }
  return res;
}

}  // namespace motis::ep
