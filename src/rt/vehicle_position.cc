#include "motis/rt/vehicle_position.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/verify.h"

namespace gtfsrt = transit_realtime;

namespace motis::vehicle_positions {

namespace {

bool valid_pos(geo::latlng const& pos) {
  return std::isfinite(pos.lat_) && std::isfinite(pos.lng_) &&
         -90.0 <= pos.lat_ && pos.lat_ <= 90.0 && -180.0 <= pos.lng_ &&
         pos.lng_ <= 180.0 && !(pos.lat_ == 0.0 && pos.lng_ == 0.0);
}

std::optional<std::string> present_string(std::string const& s) {
  return s.empty() ? std::nullopt : std::optional{s};
}

std::optional<std::string> present_enum_name(std::string const& name) {
  return name.empty() ? std::nullopt : std::optional{name};
}

}  // namespace

bool vehicle_viewport::contains(geo::latlng const& pos) const {
  auto const min_lat = std::min(min_.lat_, max_.lat_);
  auto const max_lat = std::max(min_.lat_, max_.lat_);
  auto const min_lng = std::min(min_.lng_, max_.lng_);
  auto const max_lng = std::max(min_.lng_, max_.lng_);
  return min_lat <= pos.lat_ && pos.lat_ <= max_lat && min_lng <= pos.lng_ &&
         pos.lng_ <= max_lng;
}

std::vector<vehicle_position> parse_gtfsrt_vehicle_positions(
    std::string_view const feed_id,
    gtfsrt::FeedMessage const& msg,
    std::int64_t const ingested_time) {
  auto positions = std::vector<vehicle_position>{};

  for (auto const& entity : msg.entity()) {
    if (!entity.has_vehicle() || !entity.vehicle().has_position()) {
      continue;
    }

    auto const& gtfs_vehicle = entity.vehicle();
    auto const& gtfs_pos = gtfs_vehicle.position();
    auto const pos =
        geo::latlng{gtfs_pos.latitude(), gtfs_pos.longitude()};
    if (!valid_pos(pos)) {
      continue;
    }

    auto vehicle = vehicle_position{
        .feed_id_ = std::string{feed_id},
        .entity_id_ = entity.id(),
        .reported_position_ =
            reported_position{.pos_ = pos,
                              .bearing_ = gtfs_pos.has_bearing()
                                              ? std::optional<double>{
                                                    gtfs_pos.bearing()}
                                              : std::nullopt,
                              .speed_mps_ = gtfs_pos.has_speed()
                                                ? std::optional<double>{
                                                      gtfs_pos.speed()}
                                                : std::nullopt},
        .current_stop_sequence_ =
            gtfs_vehicle.has_current_stop_sequence()
                ? std::optional<std::uint32_t>{
                      gtfs_vehicle.current_stop_sequence()}
                : std::nullopt,
        .stop_id_ = present_string(gtfs_vehicle.stop_id()),
        .current_status_ =
            gtfs_vehicle.has_current_status()
                ? present_enum_name(
                      gtfsrt::VehiclePosition_VehicleStopStatus_Name(
                          gtfs_vehicle.current_status()))
                : std::nullopt,
        .occupancy_status_ =
            gtfs_vehicle.has_occupancy_status()
                ? present_enum_name(
                      gtfsrt::VehiclePosition_OccupancyStatus_Name(
                          gtfs_vehicle.occupancy_status()))
                : std::nullopt,
        .reported_time_ = gtfs_vehicle.has_timestamp()
                              ? std::optional<std::int64_t>{
                                    static_cast<std::int64_t>(
                                        gtfs_vehicle.timestamp())}
                              : std::nullopt,
        .ingested_time_ = ingested_time};

    if (gtfs_vehicle.has_vehicle()) {
      auto const& descriptor = gtfs_vehicle.vehicle();
      vehicle.vehicle_.id_ = present_string(descriptor.id());
      vehicle.vehicle_.label_ = present_string(descriptor.label());
      vehicle.vehicle_.license_plate_ =
          present_string(descriptor.license_plate());
      if (descriptor.has_wheelchair_accessible()) {
        vehicle.vehicle_.wheelchair_accessible_ =
            gtfsrt::VehicleDescriptor_WheelchairAccessible_Name(
                descriptor.wheelchair_accessible());
      }
    }

    if (gtfs_vehicle.has_trip()) {
      auto const& trip = gtfs_vehicle.trip();
      vehicle.trip_.trip_id_ = present_string(trip.trip_id());
      vehicle.trip_.start_date_ = present_string(trip.start_date());
      vehicle.trip_.start_time_ = present_string(trip.start_time());
      vehicle.trip_.route_id_ = present_string(trip.route_id());
      if (trip.has_direction_id()) {
        vehicle.trip_.direction_id_ = trip.direction_id();
      }
      if (trip.has_schedule_relationship()) {
        vehicle.trip_.schedule_relationship_ =
            gtfsrt::TripDescriptor_ScheduleRelationship_Name(
                trip.schedule_relationship());
      }
    }

    positions.emplace_back(std::move(vehicle));
  }

  std::ranges::sort(positions, {}, [](vehicle_position const& x) {
    return std::pair{x.feed_id_, x.entity_id_};
  });
  return positions;
}

std::vector<vehicle_position> parse_gtfsrt_vehicle_positions(
    std::string_view const feed_id,
    std::string_view const protobuf_body,
    std::int64_t const ingested_time) {
  auto msg = gtfsrt::FeedMessage{};
  utl::verify(msg.ParseFromArray(protobuf_body.data(),
                                 static_cast<int>(protobuf_body.size())),
              "unable to parse GTFS-RT vehicle positions");
  return parse_gtfsrt_vehicle_positions(feed_id, msg, ingested_time);
}

void vehicle_position_store::replace_feed(
    std::string feed_id,
    std::vector<vehicle_position> positions) {
  std::erase_if(positions_, [&](vehicle_position const& pos) {
    return pos.feed_id_ == feed_id;
  });
  for (auto& pos : positions) {
    pos.feed_id_ = feed_id;
  }
  positions_.insert(end(positions_), begin(positions), end(positions));
  std::ranges::sort(positions_, {}, [](vehicle_position const& x) {
    return std::pair{x.feed_id_, x.entity_id_};
  });
}

std::vector<vehicle_position> vehicle_position_store::snapshot(
    vehicle_viewport const& viewport) const {
  auto result = std::vector<vehicle_position>{};
  for (auto const& pos : positions_) {
    if (viewport.contains(pos.reported_position_.pos_)) {
      result.emplace_back(pos);
    }
  }
  std::ranges::sort(result, {}, [](vehicle_position const& x) {
    return std::pair{x.feed_id_, x.entity_id_};
  });
  return result;
}

bool vehicle_position_store::empty() const {
  return positions_.empty();
}

}  // namespace motis::vehicle_positions
