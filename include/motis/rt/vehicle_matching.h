#pragma once

#include <optional>

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/rt/vehicle_position.h"

namespace nigiri::rt {
struct frun;
}

namespace motis::vehicle_matching {

struct vehicle_details {
  api::VehicleMatchStateEnum match_state_{
      api::VehicleMatchStateEnum::UNMATCHED};
  std::optional<std::string> scheduled_trip_id_;
  std::optional<std::string> headsign_;
  std::optional<api::TransitVehicleRouteInfo> route_;
  std::optional<api::ModeEnum> mode_;
  std::optional<api::EncodedPolyline> shape_;
  std::optional<std::string> shape_id_;
  std::optional<api::VehicleShapeSourceEnum> shape_source_;
};

std::optional<nigiri::rt::frun> resolve_run(
    tag_lookup const&,
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    vehicle_positions::vehicle_position const&);

std::optional<nigiri::rt::frun> resolve_static_trip_run_by_id(
    tag_lookup const&,
    nigiri::timetable const&,
    vehicle_positions::vehicle_position const&);

std::optional<api::TransitVehicleRouteInfo> resolve_route_info_by_id(
    tag_lookup const&,
    nigiri::timetable const&,
    vehicle_positions::vehicle_position const&,
    nigiri::lang_t const&);

std::optional<api::ModeEnum> resolve_mode_by_route_id(
    tag_lookup const&,
    nigiri::timetable const&,
    vehicle_positions::vehicle_position const&);

vehicle_details resolve_details(
    tag_lookup const*,
    nigiri::timetable const*,
    nigiri::rt_timetable const*,
    nigiri::shapes_storage const*,
    vehicle_positions::vehicle_position const&,
    nigiri::lang_t const&);

api::VehiclePosition to_api(vehicle_positions::vehicle_position const&,
                            vehicle_details,
                            bool include_shapes);

std::optional<api::VehiclePosition> primary_vehicle(
    tag_lookup const&,
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    nigiri::shapes_storage const*,
    vehicle_positions::vehicle_position_store const&,
    nigiri::rt::frun const& target,
    nigiri::lang_t const&);

}  // namespace motis::vehicle_matching
