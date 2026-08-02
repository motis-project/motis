#include "motis/osr/parameters.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "utl/verify.h"

#include "osr/routing/profiles/bike.h"
#include "osr/routing/profiles/bike_sharing.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_parking.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/profiles/hgv.h"

namespace motis {

template <typename T>
concept HasPedestrianProfile =
    requires(T const& params) { params.pedestrianProfile_; };

template <typename T>
concept HasPedestrianProfileAndSpeed =
    HasPedestrianProfile<T> &&
    std::is_same_v<decltype(std::declval<T>().pedestrianSpeed_),
                   std::optional<double>>;

template <typename T>
bool use_wheelchair(T const&) {
  return false;
}

template <typename T>
bool use_wheelchair(T const& t)
  requires HasPedestrianProfile<T>
{
  return t.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR;
}

template <typename T>
float pedestrian_speed(T const&) {
  return osr_parameters::kFootSpeed;
}

template <>
float pedestrian_speed(api::PedestrianProfileEnum const& p) {
  return p == api::PedestrianProfileEnum::FOOT
             ? osr_parameters::kFootSpeed
             : osr_parameters::kWheelchairSpeed;
}

template <typename T>
float pedestrian_speed(T const& params)
  requires HasPedestrianProfile<T>
{
  return pedestrian_speed(params.pedestrianProfile_);
}

template <typename T>
float pedestrian_speed(T const& params)
  requires HasPedestrianProfileAndSpeed<T>
{
  return params.pedestrianSpeed_
      .and_then([](auto const speed) {
        return speed > 0.3 && speed < 10.0
                   ? std::optional{static_cast<float>(speed)}
                   : std::nullopt;
      })
      .value_or(pedestrian_speed(params.pedestrianProfile_));
}

template <typename T>
float cycling_speed(T const&) {
  return osr_parameters::kBikeSpeed;
}

template <typename T>
float cycling_speed(T const& params)
  requires(
      std::is_same_v<decltype(params.cyclingSpeed_), std::optional<double>>)
{
  return params.cyclingSpeed_
      .and_then([](auto const speed) {
        return speed > 0.7 && speed < 20.0
                   ? std::optional{static_cast<float>(speed)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kBikeSpeed);
}

template <typename T>
float hgv_height(T const&) {
  return osr_parameters::kHgvHeightMeters;
}

template <typename T>
float hgv_height(T const& params)
  requires requires(T const& x) {
    { x.vehicleHeight_ } -> std::same_as<std::optional<double> const&>;
  }
{
  return params.vehicleHeight_
      .and_then([](double const height) {
        return height > 0.5 && height < 20.0
                   ? std::optional{static_cast<float>(height)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvHeightMeters);
}

template <typename T>
float hgv_width(T const&) {
  return osr_parameters::kHgvWidthMeters;
}

template <typename T>
float hgv_width(T const& params)
  requires requires(T const& x) {
    { x.vehicleWidth_ } -> std::same_as<std::optional<double> const&>;
  }
{
  return params.vehicleWidth_
      .and_then([](double const width) {
        return width > 0.5 && width < 20.0
                   ? std::optional{static_cast<float>(width)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvWidthMeters);
}

template <typename T>
float hgv_length(T const&) {
  return osr_parameters::kHgvLengthMeters;
}

template <typename T>
float hgv_length(T const& params)
  requires requires(T const& x) {
    { x.vehicleLength_ } -> std::same_as<std::optional<double> const&>;
  }
{
  return params.vehicleLength_
      .and_then([](double const length) {
        return length > 1.0 && length < 100.0
                   ? std::optional{static_cast<float>(length)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvLengthMeters);
}

template <typename T>
float hgv_weight(T const&) {
  return osr_parameters::kHgvWeightTons;
}

template <typename T>
float hgv_weight(T const& params)
  requires requires(T const& x) {
    { x.vehicleWeight_ } -> std::same_as<std::optional<double> const&>;
  }
{
  return params.vehicleWeight_
      .and_then([](double const weight) {
        return weight > 0.1 && weight < 1000.0
                   ? std::optional{static_cast<float>(weight)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvWeightTons);
}

template <typename T>
bool hgv_hazmat(T const&) {
  return osr_parameters::kHgvHazmat;
}

template <typename T>
bool hgv_hazmat(T const& params)
  requires requires(T const& x) {
    { x.vehicleHazmat_ } -> std::same_as<std::optional<bool> const&>;
  }
{
  return params.vehicleHazmat_.value_or(osr_parameters::kHgvHazmat);
}

template <typename T>
bool hgv_hazmat_water(T const&) {
  return osr_parameters::kHgvHazmatWater;
}

template <typename T>
bool hgv_hazmat_water(T const& params)
  requires requires(T const& x) {
    { x.vehicleHazmatWater_ } -> std::same_as<std::optional<bool> const&>;
  }
{
  return params.vehicleHazmatWater_.value_or(osr_parameters::kHgvHazmatWater);
}

template <typename T>
std::uint8_t hgv_axle_count(T const&) {
  return osr_parameters::kHgvAxleCount;
}

template <typename T>
std::uint8_t hgv_axle_count(T const& params)
  requires requires(T const& x) {
    { x.vehicleAxleCount_ } -> std::same_as<std::optional<std::int64_t> const&>;
  }
{
  return params.vehicleAxleCount_
      .and_then([](std::int64_t const count) {
        return count > 0 && count < 256
                   ? std::optional{static_cast<std::uint8_t>(count)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvAxleCount);
}

template <typename T>
float hgv_axle_load(T const&) {
  return osr_parameters::kHgvAxleLoadTons;
}

template <typename T>
float hgv_axle_load(T const& params)
  requires requires(T const& x) {
    { x.vehicleAxleLoad_ } -> std::same_as<std::optional<double> const&>;
  }
{
  return params.vehicleAxleLoad_
      .and_then([](double const axle_load) {
        return axle_load > 0.1 && axle_load < 100.0
                   ? std::optional{static_cast<float>(axle_load)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvAxleLoadTons);
}

template <typename T>
bool hgv_trailer(T const&) {
  return osr_parameters::kHgvTrailer;
}

template <typename T>
bool hgv_trailer(T const& params)
  requires requires(T const& x) {
    { x.vehicleTrailer_ } -> std::same_as<std::optional<bool> const&>;
  }
{
  return params.vehicleTrailer_.value_or(osr_parameters::kHgvTrailer);
}

template <typename T>
std::uint8_t hgv_top_speed(T const&) {
  return osr_parameters::kHgvTopSpeedKmh;
}

template <typename T>
std::uint8_t hgv_top_speed(T const& params)
  requires requires(T const& x) {
    { x.vehicleTopSpeed_ } -> std::same_as<std::optional<std::int64_t> const&>;
  }
{
  return params.vehicleTopSpeed_
      .and_then([](std::int64_t const top_speed) {
        return top_speed > 0 && top_speed < 256
                   ? std::optional{static_cast<std::uint8_t>(top_speed)}
                   : std::nullopt;
      })
      .value_or(osr_parameters::kHgvTopSpeedKmh);
}

template <typename T>
bool hgv_low_emission_zone_access(T const&) {
  return osr_parameters::kHgvLowEmissionZoneAccess;
}

template <typename T>
bool hgv_low_emission_zone_access(T const& params)
  requires requires(T const& x) {
    { x.vehicleLezAccess_ } -> std::same_as<std::optional<bool> const&>;
  }
{
  return params.vehicleLezAccess_.value_or(
      osr_parameters::kHgvLowEmissionZoneAccess);
}

template <typename T>
osr_parameters to_osr_parameters(T const& params) {
  return {
      .pedestrian_speed_ = pedestrian_speed(params),
      .cycling_speed_ = cycling_speed(params),
      .use_wheelchair_ = use_wheelchair(params),
      .hgv_height_meters_ = hgv_height(params),
      .hgv_width_meters_ = hgv_width(params),
      .hgv_length_meters_ = hgv_length(params),
      .hgv_weight_tons_ = hgv_weight(params),
      .hgv_hazmat_ = hgv_hazmat(params),
      .hgv_hazmat_water_ = hgv_hazmat_water(params),
      .hgv_axle_count_ = hgv_axle_count(params),
      .hgv_axle_load_tons_ = hgv_axle_load(params),
      .hgv_trailer_ = hgv_trailer(params),
      .hgv_top_speed_km_h_ = hgv_top_speed(params),
      .hgv_low_emission_zone_access_ = hgv_low_emission_zone_access(params),
  };
}

std::uint16_t meters_to_centimeters(float const meters) {
  return static_cast<std::uint16_t>(
      std::clamp(std::lround(meters * 100.0F), 0L,
                 static_cast<long>(std::numeric_limits<std::uint16_t>::max())));
}

std::uint16_t tons_to_100kg(float const tons) {
  return static_cast<std::uint16_t>(
      std::clamp(std::lround(tons * 10.0F), 0L,
                 static_cast<long>(std::numeric_limits<std::uint16_t>::max())));
}

osr_parameters get_osr_parameters(api::plan_params const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(api::refreshItinerary_params const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(api::oneToAll_params const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(api::oneToMany_params const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(api::OneToManyParams const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(
    api::oneToManyIntermodal_params const& params) {
  return to_osr_parameters(params);
}

osr_parameters get_osr_parameters(
    api::OneToManyIntermodalParams const& params) {
  return to_osr_parameters(params);
}

osr::profile_parameters to_profile_parameters(osr::search_profile const p,
                                              osr_parameters const& params) {
  // Ensure correct speed is used when using default parameters
  auto const wheelchair_speed = params.use_wheelchair_
                                    ? params.pedestrian_speed_
                                    : osr_parameters::kWheelchairSpeed;
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
    case osr::search_profile::kHgv:
      return osr::hgv::parameters{
          .height_cm_ = meters_to_centimeters(params.hgv_height_meters_),
          .width_cm_ = meters_to_centimeters(params.hgv_width_meters_),
          .length_cm_ = meters_to_centimeters(params.hgv_length_meters_),
          .weight_100kg_ = tons_to_100kg(params.hgv_weight_tons_),
          .hazmat_ = params.hgv_hazmat_,
          .hazmat_water_ = params.hgv_hazmat_water_,
          .axle_count_ = params.hgv_axle_count_,
          .axle_load_100kg_ = tons_to_100kg(params.hgv_axle_load_tons_),
          .trailer_ = params.hgv_trailer_,
          .top_speed_km_h_ = params.hgv_top_speed_km_h_,
          .low_emission_zone_access_ = params.hgv_low_emission_zone_access_};
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
    case osr::search_profile::kBus: return osr::bus::parameters{};
    case osr::search_profile::kRailway: return osr::railway::parameters{};
    case osr::search_profile::kFerry: return osr::ferry::parameters{};
  }
  throw utl::fail("{} is not a valid profile", static_cast<std::uint8_t>(p));
}

}  // namespace motis
