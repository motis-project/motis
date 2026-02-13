#include "motis/osr/max_distance.h"

#include <utility>

namespace motis {

double get_max_distance(osr::search_profile const profile,
                        std::chrono::seconds const t) {
  auto seconds = static_cast<double>(t.count());
  switch (profile) {
    case osr::search_profile::kWheelchair: return seconds * 0.8;
    case osr::search_profile::kFoot: return seconds * 1.1;
    case osr::search_profile::kBikeSharing: [[fallthrough]];
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: [[fallthrough]];
    case osr::search_profile::kBikeFast: [[fallthrough]];
    case osr::search_profile::kBike: return seconds * 4.0;
    case osr::search_profile::kCar: [[fallthrough]];
    case osr::search_profile::kCarDropOff: [[fallthrough]];
    case osr::search_profile::kCarDropOffWheelchair: [[fallthrough]];
    case osr::search_profile::kCarSharing: [[fallthrough]];
    case osr::search_profile::kCarParking: [[fallthrough]];
    case osr::search_profile::kCarParkingWheelchair: return seconds * 28.0;
    case osr::search_profile::kBus: return seconds * 5.0;
    case osr::search_profile::kRailway: return seconds * 5.5;
    case osr::search_profile::kFerry: return seconds * 4.0;
  }
  std::unreachable();
}

}  // namespace motis
