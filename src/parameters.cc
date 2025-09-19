#include "motis/parameters.h"

#include <optional>
#include <type_traits>

#include "utl/verify.h"

#include "osr/routing/profiles/bike.h"
#include "osr/routing/profiles/bike_sharing.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_parking.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/profiles/foot.h"

namespace motis {

template <typename T>
concept HasPedestrianProfile =
    requires(T const& params) { params.pedestrianProfile_; };

template <typename T>
auto use_wheelchair(T const&) {
  return false;
}

template <typename T>
auto use_wheelchair(T const& t)
  requires HasPedestrianProfile<T>
{
  return t.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR;
}

template <typename T>
auto pedestrian_speed(T const&) {
  return profile_parameters::kFootSpeed;
}

template <>
auto pedestrian_speed(api::PedestrianProfileEnum const& p) {
  return p == api::PedestrianProfileEnum::FOOT
             ? profile_parameters::kFootSpeed
             : profile_parameters::kWheelchairSpeed;
}

template <typename T>
auto pedestrian_speed(T const& params)
  requires HasPedestrianProfile<T>
{
  return pedestrian_speed(params.pedestrianProfile_);
}

template <typename T>
auto pedestrian_speed(T const& params)
  requires HasPedestrianProfile<T> &&
           std::is_same_v<decltype(params.pedestrianSpeed_),
                          std::optional<double>>
{
  return params.pedestrianSpeed_
      .and_then([](auto const speed) {
        return speed > 0.0 ? std::optional{static_cast<float>(speed)}
                           : std::nullopt;
      })
      .value_or(pedestrian_speed(params.pedestrianProfile_));
}

template <typename T>
auto cycling_speed(T const&) {
  return profile_parameters::kBikeSpeed;
}

template <typename T>
auto cycling_speed(T const& params)
  requires(
      std::is_same_v<decltype(params.cyclingSpeed_), std::optional<double>>)
{
  return params.cyclingSpeed_
      .and_then([](auto const speed) {
        return speed > 0.0 ? std::optional{static_cast<float>(speed)}
                           : std::nullopt;
      })
      .value_or(profile_parameters::kBikeSpeed);
}

template <typename T>
profile_parameters parameters(T const& params) {
  return {
      .pedestrian_speed_ = pedestrian_speed(params),
      .cycling_speed_ = cycling_speed(params),
      .use_wheelchair_ = use_wheelchair(params),
  };
}

profile_parameters get_parameters(api::plan_params const& params) {
  return parameters(params);
}

profile_parameters get_parameters(api::oneToAll_params const& params) {
  return parameters(params);
}

profile_parameters get_parameters(api::oneToMany_params const& params) {
  return parameters(params);
}

osr::profile_parameters build_parameters(osr::search_profile const p,
                                         profile_parameters const& params) {
  // Ensure correct speed is used when using default parameters
  auto const wheelchair_speed = params.use_wheelchair_
                                    ? params.pedestrian_speed_
                                    : profile_parameters::kWheelchairSpeed;
  switch (p) {
    case osr::search_profile::kFoot:
      return osr::foot<false, osr::elevator_tracking>::parameters{
          .speed_meters_per_second_ = params.pedestrian_speed_};
    case osr::search_profile::kWheelchair:
      return osr::foot<true, osr::elevator_tracking>::parameters{
          .speed_meters_per_second_ = wheelchair_speed};
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
          .car_ = {}, .foot_ = {.speed_meters_per_second_ = wheelchair_speed}};
    case osr::search_profile::kCarParking:
      return osr::car_parking<false, true>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarParkingWheelchair:
      return osr::car_parking<true, true>::parameters{
          .car_ = {}, .foot_ = {.speed_meters_per_second_ = wheelchair_speed}};
    case osr::search_profile::kBikeSharing:
      return osr::bike_sharing::parameters{
          .bike_ = {.speed_meters_per_second_ = params.cycling_speed_},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
    case osr::search_profile::kCarSharing:
      return osr::car_sharing<osr::track_node_tracking>::parameters{
          .car_ = {},
          .foot_ = {.speed_meters_per_second_ = params.pedestrian_speed_}};
  }
  throw utl::fail("{} is not a valid profile", static_cast<std::uint8_t>(p));
}

}  // namespace motis
