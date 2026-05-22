#include "motis/osr/max_distance.h"

#include <utility>
#include <variant>

#include "utl/overloaded.h"

namespace motis {

double get_max_distance(osr::profile_parameters const& osr_params,
                        std::chrono::seconds const t) {
  auto seconds = static_cast<double>(t.count());
  return seconds *
         std::visit(
             utl::overloaded{
                 [](osr::foot<false, osr::noop_tracking>::parameters const&
                        params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 [](osr::foot<true, osr::noop_tracking>::parameters const&
                        params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 [](osr::foot<false, osr::elevator_tracking>::parameters const&
                        params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 [](osr::foot<true, osr::elevator_tracking>::parameters const&
                        params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 [](osr::bike<osr::bike_costing::kSafe,
                              osr::kElevationNoCost>::parameters const& params)
                     -> double { return params.speed_meters_per_second_; },
                 [](osr::bike<osr::bike_costing::kFast,
                              osr::kElevationNoCost>::parameters const& params)
                     -> double { return params.speed_meters_per_second_; },
                 [](osr::bike<osr::bike_costing::kSafe,
                              osr::kElevationLowCost>::parameters const& params)
                     -> double { return params.speed_meters_per_second_; },
                 [](osr::bike<osr::bike_costing::kSafe,
                              osr::kElevationHighCost>::parameters const&
                        params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 []([[maybe_unused]] osr::car::parameters const& params)
                     -> double { return osr_parameters::kCarSpeed; },
                 []([[maybe_unused]] osr::car_parking<
                     false, false>::parameters const& params) -> double {
                   return osr_parameters::kCarSpeed;
                 },
                 []([[maybe_unused]] osr::car_parking<
                     true, false>::parameters const& params) -> double {
                   return osr_parameters::kCarSpeed;
                 },
                 []([[maybe_unused]] osr::car_parking<
                     false, true>::parameters const& params) -> double {
                   return osr_parameters::kCarSpeed;
                 },
                 []([[maybe_unused]] osr::car_parking<
                     true, true>::parameters const& params) -> double {
                   return osr_parameters::kCarSpeed;
                 },
                 [](osr::bike_sharing::parameters const& params) -> double {
                   return params.bike_.speed_meters_per_second_;
                 },
                 []([[maybe_unused]] osr::car_sharing<
                     osr::track_node_tracking>::parameters const& params)
                     -> double { return osr_parameters::kCarSpeed; },
                 []([[maybe_unused]] osr::bus::parameters const& params)
                     -> double { return osr_parameters::kBusSpeed; },
                 []([[maybe_unused]] osr::railway::parameters const& params)
                     -> double { return osr_parameters::kRailwaySpeed; },
                 []([[maybe_unused]] osr::ferry::parameters const& params)
                     -> double { return osr_parameters::kFerrySpeed; }},
             osr_params);
}

double get_max_distance(osr::search_profile const profile,
                        std::chrono::seconds const t) {
  auto seconds = static_cast<double>(t.count());
  switch (profile) {
    case osr::search_profile::kWheelchair:
      return seconds * osr_parameters::kWheelchairSpeed;
    case osr::search_profile::kFoot:
      return seconds * osr_parameters::kFootSpeed;
    case osr::search_profile::kBikeSharing: [[fallthrough]];
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: [[fallthrough]];
    case osr::search_profile::kBikeFast: [[fallthrough]];
    case osr::search_profile::kBike:
      return seconds * osr_parameters::kBikeSpeed;
    case osr::search_profile::kCar: [[fallthrough]];
    case osr::search_profile::kCarDropOff: [[fallthrough]];
    case osr::search_profile::kCarDropOffWheelchair: [[fallthrough]];
    case osr::search_profile::kCarSharing: [[fallthrough]];
    case osr::search_profile::kCarParking: [[fallthrough]];
    case osr::search_profile::kCarParkingWheelchair:
      return seconds * osr_parameters::kCarSpeed;
    case osr::search_profile::kBus: return seconds * osr_parameters::kBusSpeed;
    case osr::search_profile::kRailway:
      return seconds * osr_parameters::kRailwaySpeed;
    case osr::search_profile::kFerry:
      return seconds * osr_parameters::kFerrySpeed;
  }
  std::unreachable();
}

double get_max_distance(osr::search_profile const profile,
                        osr_parameters const& osr_params,
                        std::chrono::seconds const t) {
  return get_max_distance(to_profile_parameters(profile, osr_params), t);
}

}  // namespace motis
