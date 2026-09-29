#include "motis/osr/mode_to_profile.h"

#include "utl/verify.h"

namespace motis {

api::ModeEnum to_mode(osr::search_profile const m) {
  switch (m) {
    case osr::search_profile::kCarParkingWheelchair: [[fallthrough]];
    case osr::search_profile::kCarParking: return api::ModeEnum::CAR_PARKING;
    case osr::search_profile::kCarDropOffWheelchair: [[fallthrough]];
    case osr::search_profile::kCarDropOff: return api::ModeEnum::CAR_DROPOFF;
    case osr::search_profile::kFoot: [[fallthrough]];
    case osr::search_profile::kWheelchair: return api::ModeEnum::WALK;
    case osr::search_profile::kCar: return api::ModeEnum::CAR;
    case osr::search_profile::kHgv: return api::ModeEnum::HGV;
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: [[fallthrough]];
    case osr::search_profile::kBikeFast: [[fallthrough]];
    case osr::search_profile::kBike: return api::ModeEnum::BIKE;
    case osr::search_profile::kBikeSharing: [[fallthrough]];
    case osr::search_profile::kCarSharing: return api::ModeEnum::RENTAL;
    case osr::search_profile::kBus: return api::ModeEnum::DEBUG_BUS_ROUTE;
    case osr::search_profile::kRailway:
      return api::ModeEnum::DEBUG_RAILWAY_ROUTE;
    case osr::search_profile::kFerry: return api::ModeEnum::DEBUG_FERRY_ROUTE;
  }
  std::unreachable();
}

api::ModeEnum to_mode(osr::mode const m) {
  switch (m) {
    case osr::mode::kFoot: [[fallthrough]];
    case osr::mode::kWheelchair: return api::ModeEnum::WALK;
    case osr::mode::kBike: return api::ModeEnum::BIKE;
    case osr::mode::kCar: return api::ModeEnum::CAR;
    case osr::mode::kHgv: return api::ModeEnum::HGV;
    case osr::mode::kRailway: return api::ModeEnum::DEBUG_RAILWAY_ROUTE;
    case osr::mode::kFerry: return api::ModeEnum::DEBUG_FERRY_ROUTE;
  }
  std::unreachable();
}

osr::search_profile to_profile(
    api::ModeEnum const m,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs) {
  auto const wheelchair =
      pedestrian_profile == api::PedestrianProfileEnum::WHEELCHAIR;
  switch (m) {
    case api::ModeEnum::WALK:
      return wheelchair ? osr::search_profile::kWheelchair
                        : osr::search_profile::kFoot;
    case api::ModeEnum::BIKE:
      switch (elevation_costs) {
        case api::ElevationCostsEnum::NONE: return osr::search_profile::kBike;
        case api::ElevationCostsEnum::LOW:
          return osr::search_profile::kBikeElevationLow;
        case api::ElevationCostsEnum::HIGH:
          return osr::search_profile::kBikeElevationHigh;
      }
      return osr::search_profile::kBike;  // Fallback if invalid value is used
    case api::ModeEnum::ODM: [[fallthrough]];
    case api::ModeEnum::RIDE_SHARING: [[fallthrough]];
    case api::ModeEnum::CAR: return osr::search_profile::kCar;
    case api::ModeEnum::HGV: return osr::search_profile::kHgv;
    case api::ModeEnum::CAR_DROPOFF:
      return wheelchair ? osr::search_profile::kCarDropOffWheelchair
                        : osr::search_profile::kCarDropOff;
    case api::ModeEnum::CAR_PARKING:
      return wheelchair ? osr::search_profile::kCarParkingWheelchair
                        : osr::search_profile::kCarParking;
    case api::ModeEnum::RENTAL:
      // could be kBikeSharing or kCarSharing, use gbfs::get_osr_profile()
      // to get the correct profile for each product
      return osr::search_profile::kBikeSharing;
    case api::ModeEnum::DEBUG_BUS_ROUTE: return osr::search_profile::kBus;
    case api::ModeEnum::DEBUG_RAILWAY_ROUTE:
      return osr::search_profile::kRailway;
    case api::ModeEnum::DEBUG_FERRY_ROUTE: return osr::search_profile::kFerry;
    default: throw utl::fail("unsupported mode");
  }
}

}  // namespace motis
