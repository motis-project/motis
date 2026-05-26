#include "motis/osr/max_distance.h"

#include <chrono>
#include <concepts>
#include <utility>
#include <variant>

#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_parking.h"

namespace motis {

// Cannot use utl::overloaded instead
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename T>
concept HasSpeed = osr::ProfileParameters<T> && requires(T t) {
  { t.speed_meters_per_second_ };
};

template <typename T>
concept IsCarProfileParameters =
    osr::ProfileParameters<T> && !HasSpeed<T> && requires {
      std::same_as<T, osr::car::parameters> ||
          std::same_as<T, osr::car_parking<true, true>::parameters> ||
          std::same_as<T, osr::car_parking<true, false>::parameters> ||
          std::same_as<T, osr::car_parking<false, true>::parameters> ||
          std::same_as<T, osr::car_parking<false, false>::parameters>;
    };

constexpr double get_max_distance(osr::profile_parameters const& osr_params,
                                  std::chrono::seconds const t) {
  auto seconds = static_cast<double>(t.count());
  return seconds *
         std::visit(
             overloaded{
                 []<HasSpeed Params>(Params const& params) -> double {
                   return params.speed_meters_per_second_;
                 },
                 []<IsCarProfileParameters Params>(Params const&) -> double {
                   return osr_parameters::kCarSpeed;
                 },
                 [](osr::bus::parameters const&) -> double {
                   return osr_parameters::kBusSpeed;
                 },
                 [](osr::railway::parameters const&) -> double {
                   return osr_parameters::kRailwaySpeed;
                 },
                 [](osr::ferry::parameters const&) -> double {
                   return osr_parameters::kFerrySpeed;
                 },
             },
             osr_params);
}

static_assert(get_max_distance(osr::foot<true>::parameters{2.1f},
                               std::chrono::seconds(1)) == 2.1f);
static_assert(
    get_max_distance(
        osr::bike<osr::bike_costing::kFast, osr::kElevationNoCost>::parameters{
            7.2f},
        std::chrono::seconds(2)) == 14.4f);
static_assert(get_max_distance(osr::car::parameters{},
                               std::chrono::seconds(3)) ==
              3 * osr_parameters::kCarSpeed);
static_assert(get_max_distance(osr::car_parking<false, true>::parameters{},
                               std::chrono::seconds(4)) ==
              4 * osr_parameters::kCarSpeed);
static_assert(get_max_distance(osr::bus::parameters{},
                               std::chrono::seconds(8)) ==
              8 * osr_parameters::kBusSpeed);
static_assert(get_max_distance(osr::railway::parameters{},
                               std::chrono::seconds(16)) ==
              16 * osr_parameters::kRailwaySpeed);
static_assert(get_max_distance(osr::ferry::parameters{},
                               std::chrono::seconds(32)) ==
              32 * osr_parameters::kFerrySpeed);

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
