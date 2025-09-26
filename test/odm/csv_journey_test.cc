#include "gtest/gtest.h"

#include "motis/odm/journeys.h"
#include "motis/odm/odm.h"

using namespace std::string_view_literals;
using namespace date;
using namespace std::chrono_literals;
using namespace motis::odm;

constexpr auto const csv0 =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 12:34, 2025-06-16 23:45, 3, taxi, 12, walk, 04
2025-06-17 11:11, 2025-06-17 22:22, 2, walk, 05, walk, 06
2025-06-18 06:33, 2025-06-18 07:44, 8, taxi, 20, taxi, 20
)__"sv;

TEST(odm, csv_journeys_in_out) { EXPECT_EQ(csv0, to_csv(from_csv(csv0))); }

constexpr auto const direct_odm_csv =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 12:00, 2025-06-16 13:00, 0, taxi, 60, walk, 00
)__"sv;

TEST(odm, csv_journeys_direct) {
  EXPECT_EQ(
      direct_odm_csv,
      to_csv(std::vector<nigiri::routing::journey>{make_odm_direct(
          nigiri::location_idx_t::invalid(), nigiri::location_idx_t::invalid(),
          nigiri::unixtime_t{date::sys_days{2025_y / June / 16} + 12h},
          nigiri::unixtime_t{date::sys_days{2025_y / June / 16} + 13h})}));
}