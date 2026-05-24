#include "boost/url/url_view.hpp"

#include "gtest/gtest.h"

#include "motis/endpoints/nearest.h"

#include "geo/latlng.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"
#include "osr/routing/with_profile.h"

static std::string query(osr::search_profile const p,
                         osr::location const loc,
                         int const n,
                         int const r) {
  return fmt::format("/nearest/v1/{}/{},{}?number={}&radiuses={}",
                     osr::to_str(p), loc.pos_.lng(), loc.pos_.lat(), n, r);
}

TEST(motis, nearest_endpoint_profiles) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = motis::config{.osm_ = {"test/resources/test_case.osm.pbf"},
                               .street_routing_ = true};
  motis::import(c, "test/data");
  auto d = motis::data{"test/data", c};
  auto const ep = motis::ep::nearest{*d.w_, *d.l_, c};

  auto const loc = osr::location{{49.87260, 8.63085}, osr::kNoLevel};
  auto const radius = 100;
  auto const number = 100;

  for (auto i = 0U; i <= static_cast<uint8_t>(osr::search_profile::kFerry);
       ++i) {
    auto const profile = static_cast<osr::search_profile>(i);
    for (auto n = 1U; n <= number; ++n) {
      auto const q = query(profile, loc, n, radius);
      auto const res = ep(boost::urls::url_view{q});

      auto const expected = osr::with_profile(profile, [&]<typename P>(P&&) {
        return d.l_->match<P>(
            std::get<typename P::parameters>(osr::get_parameters(profile)), loc,
            false, osr::direction::kForward, radius, nullptr);
      });

      if (res.code_ == "NoSegment") {
        EXPECT_TRUE(expected.empty());
        continue;
      }

      auto const size = std::min(static_cast<std::size_t>(n), expected.size());
      ASSERT_EQ(res.waypoints_.size(), size);

      for (auto i = 0U; i < size; ++i) {
        EXPECT_EQ((*res.waypoints_[i].location_)[0],
                  expected[i].closest_point_on_way_.lng());
        EXPECT_EQ((*res.waypoints_[i].location_)[1],
                  expected[i].closest_point_on_way_.lat());
        EXPECT_EQ(res.waypoints_[i].distance_,
                  geo::distance(expected[i].closest_point_on_way_, loc.pos_));
      }
    }
  }
}

TEST(motis, nearest_endpoint_invalid) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = motis::config{.osm_ = {"test/resources/test_case.osm.pbf"},
                               .street_routing_ = true};
  motis::import(c, "test/data");
  auto d = motis::data{"test/data", c};
  auto const ep = motis::ep::nearest{*d.w_, *d.l_, c};

  EXPECT_EQ(ep(boost::urls::url_view{"/nearest/v1/foot"}).code_, "InvalidUrl");

  EXPECT_EQ(ep(boost::urls::url_view{"/nearest/v1/foot/abc,def"}).code_,
            "InvalidQuery");

  auto const bad_number =
      query(osr::search_profile::kFoot,
            osr::location{{49.8731904, 8.6221451}, osr::kNoLevel}, 0, 100);
  EXPECT_EQ(ep(boost::urls::url_view{bad_number}).code_, "InvalidValue");
}
