#include "motis/parameters.h"

#include "osr/routing/profiles/bike.h"
#include "osr/routing/profiles/bike_sharing.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_parking.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/profiles/foot.h"

namespace motis {

profile_parameters get_parameters(
    api::PedestrianProfileEnum const pedestrian_profile,
    std::optional<api::PedestrianSpeed> const& p_speed,
    std::optional<api::CyclingSpeed> const& b_speed) {
  return {
      .pedestrian_speed_ =
          p_speed
              .and_then([](auto const speed) {
                return speed > 0.0 ? std::optional{static_cast<float>(speed)}
                                   : std::nullopt;
              })
              .value_or(pedestrian_profile == api::PedestrianProfileEnum::FOOT
                            ? profile_parameters::kFootSpeed
                            : profile_parameters::kWheelchairSpeed),

      .cycling_speed_ =
          b_speed
              .and_then([](auto const speed) {
                return speed > 0.0 ? std::optional{static_cast<float>(speed)}
                                   : std::nullopt;
              })
              .value_or(profile_parameters::kBikeSpeed)};
}

osr::profile_parameters build_parameters(osr::search_profile const p,
                                         profile_parameters const& params) {
  // api::PedestrianSpeed const p_speed,
  // api::CyclingSpeed const b_speed) {
  // constexpr auto const kFootSpeed = 1.2F;
  // constexpr auto const kWheelchairSpeed = 0.8F;
  // constexpr auto const kBikeSpeed = 4.2F;

  switch (p) {
    case osr::search_profile::kFoot:
      return osr::foot<false, osr::elevator_tracking>::parameters{
          .speed_meters_per_second_ = params.pedestrian_speed_};
    case osr::search_profile::kWheelchair:
      return osr::foot<true, osr::elevator_tracking>::parameters{
          .speed_meters_per_second_ = params.pedestrian_speed_};
    case osr::search_profile::kBike:
      return osr::bike<osr::bike_costing::kSafe,
                       osr::kElevationNoCost>::parameters{
          .speed_meters_per_second_ = params.cycling_speed_};
    case osr::search_profile::kBikeFast:
      return osr::bike<osr::bike_costing::kFast,
                       osr::kElevationNoCost>::parameters{
          .speed_meters_per_second_ = params.cycling_speed_};
    case osr::search_profile::kBikeElevationLow:
      return osr::bike<osr::bike_costing::kSafe,
                       osr::kElevationLowCost>::parameters{
          .speed_meters_per_second_ = params.cycling_speed_};
    case osr::search_profile::kBikeElevationHigh:
      return osr::bike<osr::bike_costing::kSafe,
                       osr::kElevationHighCost>::parameters{
          .speed_meters_per_second_ = params.cycling_speed_};
    case osr::search_profile::kCar: return osr::car::parameters{};
    case osr::search_profile::kCarDropOff:
      return osr::car_parking<false, false>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarDropOffWheelchair:
      return osr::car_parking<true, false>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarParking:
      return osr::car_parking<false, true>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarParkingWheelchair:
      return osr::car_parking<true, true>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kBikeSharing:
      return osr::bike_sharing::parameters{
          .bike_ = {.speed_meters_per_second_ = params.cycling_speed_},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarSharing:
      return osr::car_sharing<osr::track_node_tracking>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
  }
  // switch (p) {
  //   case osr::search_profile::kFoot:
  //     return osr::foot<false, osr::elevator_tracking>::parameters{
  //         .speed_meters_per_second_ =
  //             p_speed > 0 ? static_cast<float>(p_speed) : kFootSpeed};
  //   case osr::search_profile::kWheelchair:
  //     return osr::foot<true, osr::elevator_tracking>::parameters{
  //         .speed_meters_per_second_ =
  //             p_speed > 0 ? static_cast<float>(p_speed) :
  //             kWheelchairSpeed};
  //   case osr::search_profile::kBike:
  //     return osr::bike<osr::bike_costing::kSafe,
  //                      osr::kElevationNoCost>::parameters{
  //         .speed_meters_per_second_ =
  //             b_speed > 0 ? static_cast<float>(b_speed) : kBikeSpeed};
  //   case osr::search_profile::kBikeFast:
  //     return osr::bike<osr::bike_costing::kFast,
  //                      osr::kElevationNoCost>::parameters{
  //         .speed_meters_per_second_ =
  //             b_speed > 0 ? static_cast<float>(b_speed) : kBikeSpeed};
  //   case osr::search_profile::kBikeElevationLow:
  //     return osr::bike<osr::bike_costing::kSafe,
  //                      osr::kElevationLowCost>::parameters{
  //         .speed_meters_per_second_ =
  //             b_speed > 0 ? static_cast<float>(b_speed) : kBikeSpeed};
  //   case osr::search_profile::kBikeElevationHigh:
  //     return osr::bike<osr::bike_costing::kSafe,
  //                      osr::kElevationHighCost>::parameters{
  //         .speed_meters_per_second_ =
  //             b_speed > 0 ? static_cast<float>(b_speed) : kBikeSpeed};
  //   case osr::search_profile::kCar: return osr::car::parameters{};
  //   case osr::search_profile::kCarDropOff:
  //     return osr::car_parking<false, false>::parameters{
  //         .car_ = {},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kFootSpeed}};
  //   case osr::search_profile::kCarDropOffWheelchair:
  //     return osr::car_parking<true, false>::parameters{
  //         .car_ = {},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kWheelchairSpeed}};
  //   case osr::search_profile::kCarParking:
  //     return osr::car_parking<false, true>::parameters{
  //         .car_ = {},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kFootSpeed}};
  //   case osr::search_profile::kCarParkingWheelchair:
  //     return osr::car_parking<true, true>::parameters{
  //         .car_ = {},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kWheelchairSpeed}};
  //   case osr::search_profile::kBikeSharing:
  //     return osr::bike_sharing::parameters{
  //         .bike_ = {.speed_meters_per_second_ =
  //                       b_speed > 0 ? static_cast<float>(b_speed) :
  //                       kBikeSpeed},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kFootSpeed}};
  //   case osr::search_profile::kCarSharing:
  //     return osr::car_sharing<osr::track_node_tracking>::parameters{
  //         .car_ = {},
  //         .foot_ = {.speed_meters_per_second_ =
  //                       p_speed > 0 ? static_cast<float>(p_speed)
  //                                   : kFootSpeed}};
  // }
}

}  // namespace motis
