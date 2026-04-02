#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstddef>

#include "boost/json.hpp"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/endpoints/map/routes.h"
#include "motis/gbfs/update.h"

#include "../test_case.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace testing;
namespace n = nigiri;

TEST(motis, map_routes) {
  auto& d = get_test_case<test_case::FFM_simple_transfers>();

  auto const map_routes = utl::init_from<ep::routes>(d).value();

  {
    auto const res = map_routes(
        "/api/experimental/map/routes"
        "?max=49.88135900212875%2C8.60917200508915"
        "&min=49.863844157325886%2C8.649823169526556"
        "&zoom=16");
    EXPECT_EQ(res.routes_.size(), 2U);
    EXPECT_EQ(res.zoomFiltered_, false);

    EXPECT_THAT(res.routes_, Contains(Field(&api::RouteInfo::mode_,
                                            Eq(api::ModeEnum::BUS))));
    EXPECT_THAT(res.routes_, Contains(Field(&api::RouteInfo::mode_,
                                            Eq(api::ModeEnum::TRAM))));
    EXPECT_THAT(res.routes_, Each(Field(&api::RouteInfo::pathSource_,
                                        Eq(api::RoutePathSourceEnum::ROUTED))));
    EXPECT_FALSE(res.polylines_.empty());
    EXPECT_FALSE(res.stops_.empty());

    for (auto const& route : res.routes_) {
      for (auto const& segment : route.segments_) {
        EXPECT_GE(segment.from_, 0);
        EXPECT_GE(segment.to_, 0);
        EXPECT_GE(segment.polyline_, 0);
        EXPECT_LT(segment.from_, static_cast<std::int64_t>(res.stops_.size()));
        EXPECT_LT(segment.to_, static_cast<std::int64_t>(res.stops_.size()));
        EXPECT_LT(segment.polyline_,
                  static_cast<std::int64_t>(res.polylines_.size()));
      }
    }

    for (auto route_index = 0U; route_index != res.routes_.size();
         ++route_index) {
      for (auto const& segment : res.routes_[route_index].segments_) {
        EXPECT_THAT(
            res.polylines_.at(static_cast<std::size_t>(segment.polyline_))
                .routeIndexes_,
            Contains(static_cast<std::int64_t>(route_index)));
      }
    }
  }

  {
    // map section without data
    auto const res = map_routes(
        "/api/experimental/map/routes"
        "?max=53.5757876577963%2C9.904453881311966"
        "&min=53.518462458295005%2C10.04877290275494"
        "&zoom=14.5");
    EXPECT_EQ(res.routes_.size(), 0U);
    EXPECT_EQ(res.zoomFiltered_, false);
  }
}
