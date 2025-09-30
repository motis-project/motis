#include "gtest/gtest.h"

#include "motis/endpoints/routing.h"

using namespace std::chrono_literals;
using namespace date;
using namespace motis::ep;
namespace n = nigiri;
using iv = n::interval<n::unixtime_t>;

TEST(motis, shrink) {
  auto const d = date::sys_days{2025_y / September / 29};
  {
    auto const j = [](std::uint8_t const transfers, n::unixtime_t const dep,
                      n::unixtime_t const arr) {
      auto x = n::routing::journey{};
      x.start_time_ = dep;
      x.dest_time_ = arr;
      x.transfers_ = transfers;
      return x;
    };

    auto journeys = std::vector<n::routing::journey>{
        j(2, d + 10h, d + 11h),  //
        j(1, d + 10h, d + 12h),  //
        j(0, d + 10h, d + 13h),  //
        j(2, d + 20h, d + 21h),  //
        j(1, d + 20h, d + 22h),  //
        j(0, d + 20h, d + 23h),  //
    };

    auto const i4 = shrink(false, 4, {d + 11h, d + 20h}, journeys);
    EXPECT_EQ(6, journeys.size());
    EXPECT_EQ((iv{d + 11h, d + 20h}), i4);

    auto const i3 = shrink(false, 3, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ(3, journeys.size());
    EXPECT_EQ((iv{d + 11h, d + 20h}), i3);

    auto const i2 = shrink(false, 2, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ(3, journeys.size());
    EXPECT_EQ((iv{d + 11h, d + 13h}), i2);

    auto const i1 = shrink(false, 1, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ(3, journeys.size());
    EXPECT_EQ((iv{d + 11h, d + 13h}), i1);
  }

  {
    auto const j = [](std::uint8_t const transfers, n::unixtime_t const dep,
                      n::unixtime_t const arr) {
      auto x = n::routing::journey{};
      x.start_time_ = arr;
      x.dest_time_ = dep;
      x.transfers_ = transfers;
      return x;
    };

    auto journeys = std::vector<n::routing::journey>{
        j(2, d + 10h, d + 11h),
        j(1, d + 10h, d + 12h),
        j(0, d + 10h, d + 13h),
    };

    auto const i3 = shrink(true, 3, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ(3, journeys.size());
    EXPECT_EQ((iv{d + 11h, d + 13h}), i3);

    auto const i2 = shrink(true, 2, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ((std::vector<n::routing::journey>{
                  j(1, d + 10h, d + 12h),
                  j(0, d + 10h, d + 13h),
              }),
              journeys);
    EXPECT_EQ(2, journeys.size());
    EXPECT_EQ((iv{d + 11h + 1min, d + 13h}), i2);

    auto const i1 = shrink(true, 1, {d + 11h, d + 13h}, journeys);
    EXPECT_EQ((std::vector<n::routing::journey>{
                  j(0, d + 10h, d + 13h),
              }),
              journeys);
    EXPECT_EQ(1, journeys.size());
    EXPECT_EQ((iv{d + 12h + 1min, d + 13h}), i1);
  }
}
