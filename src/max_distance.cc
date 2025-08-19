#include "motis/max_distance.h"

#include <utility>

namespace motis {

double get_max_distance(osr::search_profile const profile,
                        std::chrono::seconds const t) {
  auto seconds = static_cast<double>(t.count());
  switch (profile) {
    case osr::search_profile::kWheelchair: return seconds * 0.8;
    case osr::search_profile::kFoot: return seconds * 1.1;
    case osr::search_profile::kBike:
    case osr::search_profile::kBikeFast:
    case osr::search_profile::kBikeSharing:
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: return seconds * 4.1;
    case osr::search_profile::kCar:
    case osr::search_profile::kCarDropOff:
    case osr::search_profile::kCarDropOffWheelchair:
    case osr::search_profile::kCarSharing:
    case osr::search_profile::kCarParking: [[fallthrough]];
    case osr::search_profile::kCarParkingWheelchair: return seconds * 28.0;
  }
  std::unreachable();
}

}  // namespace motis
