#include "motis/osr/max_distance.h"

#include <chrono>
#include <concepts>
#include <utility>
#include <variant>

#include "osr/routing/parameters.h"
#include "osr/routing/profiles/bike.h"
#include "osr/routing/profiles/bike_sharing.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_parking.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/profiles/ferry.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/profiles/railway.h"
#include "osr/routing/tracking.h"

namespace motis {

namespace {

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
struct is_car_profile : std::false_type {};

template <>
struct is_car_profile<osr::car> : std::true_type {};

template <bool A, bool B>
struct is_car_profile<osr::car_parking<A, B>> : std::true_type {};

template <typename A>
struct is_car_profile<osr::car_sharing<A>> : std::true_type {};

template <typename T>
concept IsCarProfile = is_car_profile<typename T::profile_t>::value;

constexpr double get_max_distance(osr::profile_parameters const& p,
                                  std::chrono::seconds const t) {
  return static_cast<double>(t.count()) *
         std::visit(overloaded{
                        []<HasSpeed P>(P const& params) {
                          return params.speed_meters_per_second_;
                        },
                        []<IsCarProfile P>(P const&) {
                          return osr_parameters::kCarSpeed;
                        },
                        [](osr::bike_sharing::parameters const& params) {
                          return params.bike_.speed_meters_per_second_;
                        },
                        [](osr::bus::parameters const&) {
                          return osr_parameters::kBusSpeed;
                        },
                        [](osr::railway::parameters const&) {
                          return osr_parameters::kRailwaySpeed;
                        },
                        [](osr::ferry::parameters const&) {
                          return osr_parameters::kFerrySpeed;
                        },
                    },
                    p);
}

}  // namespace

double get_max_distance(osr::search_profile const profile,
                        osr_parameters const& osr_params,
                        std::chrono::seconds const t) {
  return get_max_distance(to_profile_parameters(profile, osr_params), t);
}

}  // namespace motis

// TESTS

using namespace motis;

static_assert(get_max_distance(osr::foot<true>::parameters{2.1F},
                               std::chrono::seconds(1)) == 2.1F);
static_assert(
    get_max_distance(
        osr::bike<osr::bike_costing::kFast, osr::kElevationNoCost>::parameters{
            7.2F},
        std::chrono::seconds(2)) == 14.4F);
static_assert(get_max_distance(
                  osr::bike_sharing::parameters{
                      osr::bike_sharing::bikep::parameters{8.0F},
                      osr::bike_sharing::footp::parameters{1.6F}},
                  std::chrono::seconds(5)) == 40.0);
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
