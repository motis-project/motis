#include "gtest/gtest.h"

#include "motis/odm/mixer/journeys.h"

using namespace std::string_view_literals;
using namespace std::chrono_literals;
using namespace motis::odm;

constexpr auto const csv0 =
    R"__(departure_time, arrival_time, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
12:34, 23:45, 3, taxi, 12, walk, 04
11:11, 22:22, 2, walk, 05, walk, 06
66:33, 77:44, 8, taxi, 20, taxi, 20
)__"sv;

TEST(csv_journeys, in_out) { EXPECT_EQ(csv0, to_csv(from_csv(csv0))); }

constexpr auto const direct_odm_csv =
    R"__(departure_time, arrival_time, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
12:00, 13:00, 0, taxi, 60, walk, 00
)__"sv;

TEST(csv_journeys, direct_odm) {
  EXPECT_EQ(
      direct_odm_csv,
      to_csv(std::vector<nigiri::routing::journey>{make_odm_direct(
          nigiri::location_idx_t::invalid(), nigiri::location_idx_t::invalid(),
          nigiri::unixtime_t{12h}, nigiri::unixtime_t{13h})}));
}