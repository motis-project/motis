#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "geo/latlng.h"

namespace transit_realtime {
class FeedMessage;
}  // namespace transit_realtime

namespace motis::vehicle_positions {

struct vehicle_descriptor {
  std::optional<std::string> id_;
  std::optional<std::string> label_;
  std::optional<std::string> license_plate_;
  std::optional<std::string> wheelchair_accessible_;
};

struct trip_descriptor {
  std::optional<std::string> trip_id_;
  std::optional<std::string> start_date_;
  std::optional<std::string> start_time_;
  std::optional<std::string> route_id_;
  std::optional<std::uint32_t> direction_id_;
  std::optional<std::string> schedule_relationship_;
};

struct reported_position {
  geo::latlng pos_;
  std::optional<double> bearing_;
  std::optional<double> speed_mps_;
};

struct vehicle_position {
  std::string feed_id_;
  std::string entity_id_;
  vehicle_descriptor vehicle_;
  trip_descriptor trip_;
  reported_position reported_position_;
  std::optional<std::uint32_t> current_stop_sequence_;
  std::optional<std::string> stop_id_;
  std::optional<std::string> current_status_;
  std::optional<std::string> occupancy_status_;
  std::optional<std::int64_t> reported_time_;
  std::int64_t ingested_time_{};
};

struct vehicle_viewport {
  geo::latlng min_;
  geo::latlng max_;

  bool contains(geo::latlng const&) const;
};

std::vector<vehicle_position> parse_gtfsrt_vehicle_positions(
    std::string_view feed_id,
    transit_realtime::FeedMessage const&,
    std::int64_t ingested_time);

std::vector<vehicle_position> parse_gtfsrt_vehicle_positions(
    std::string_view feed_id,
    std::string_view protobuf_body,
    std::int64_t ingested_time);

struct vehicle_position_store {
  void replace_feed(std::string feed_id, std::vector<vehicle_position>);

  [[nodiscard]] std::vector<vehicle_position> snapshot(
      vehicle_viewport const&) const;

  [[nodiscard]] bool empty() const;

private:
  std::vector<vehicle_position> positions_;
};

}  // namespace motis::vehicle_positions
