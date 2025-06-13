#include "gtest/gtest.h"

#include "motis/odm/mixer/journeys.h"
#include "motis/odm/mixer/mixer.h"
#include "motis/odm/odm.h"

using namespace std::string_view_literals;
using namespace motis::odm;

TEST(odm, tally) {
  auto const ct = std::vector<cost_threshold>{{0, 30}, {1, 1}, {10, 2}};
  EXPECT_EQ(0, tally(0, ct));
  EXPECT_EQ(30, tally(1, ct));
  EXPECT_EQ(43, tally(12, ct));
}

std::string mix(std::string_view csv) {
  auto odm_journeys = from_csv(csv);
  auto pt_journeys = nigiri::pareto_set<nigiri::routing::journey>{};
  for (auto j = begin(odm_journeys); j != end(odm_journeys);) {
    if (is_pure_pt(*j)) {
      pt_journeys.add(std::move(*j));
      j = odm_journeys.erase(j);
    } else {
      ++j;
    }
  }
  static auto const m = get_default_mixer();
  m.mix(pt_journeys, odm_journeys, nullptr);
  return to_csv(odm_journeys);
}

constexpr auto const pt_taxi_no_direct_in =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 10:10, 2025-06-16 10:20, 0, taxi, 10, walk, 00
2025-06-16 10:17, 2025-06-16 10:27, 0, taxi, 10, walk, 00
2025-06-16 10:17, 2025-06-16 11:00, 0, walk, 30, walk, 00
2025-06-16 10:43, 2025-06-16 10:53, 0, taxi, 10, walk, 00
2025-06-16 10:43, 2025-06-16 11:00, 0, taxi, 04, walk, 00
2025-06-16 10:50, 2025-06-16 11:00, 0, taxi, 10, walk, 00
2025-06-16 11:00, 2025-06-16 11:10, 0, taxi, 10, walk, 00
)__"sv;

constexpr auto const pt_taxi_no_direct_out =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 10:17, 2025-06-16 11:00, 0, walk, 30, walk, 00
2025-06-16 10:43, 2025-06-16 11:00, 0, taxi, 04, walk, 00
)__"sv;

TEST(odm, pt_taxi_no_direct) {
  EXPECT_EQ(pt_taxi_no_direct_out, mix(pt_taxi_no_direct_in));
}

constexpr auto const taxi_saves_transfers_in =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 10:00, 2025-06-16 11:00, 4, walk, 05, walk, 05
2025-06-16 10:14, 2025-06-16 11:00, 2, taxi, 06, walk, 05
2025-06-16 10:20, 2025-06-16 11:00, 1, taxi, 10, walk, 05
2025-06-16 10:30, 2025-06-16 11:00, 0, taxi, 15, walk, 05
)__"sv;

constexpr auto const taxi_saves_transfers_out =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 10:00, 2025-06-16 11:00, 4, walk, 05, walk, 05
)__"sv;

TEST(odm, taxi_saves_transfers) {
  EXPECT_EQ(taxi_saves_transfers_out, mix(taxi_saves_transfers_in));
}

constexpr auto const schleife_kleinpriebus_montag_in =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 02:00, 2025-06-16 02:41, 0, taxi, 41, walk, 00
2025-06-16 02:29, 2025-06-16 04:45, 1, walk, 09, walk, 03
2025-06-16 02:29, 2025-06-16 03:37, 0, walk, 09, taxi, 27
2025-06-16 03:00, 2025-06-16 03:41, 0, taxi, 41, walk, 00
2025-06-16 03:39, 2025-06-16 04:45, 0, taxi, 17, walk, 03
2025-06-16 04:00, 2025-06-16 04:41, 0, taxi, 41, walk, 00
2025-06-16 04:14, 2025-06-16 05:22, 1, walk, 09, taxi, 27
2025-06-16 04:29, 2025-06-16 05:37, 0, walk, 09, taxi, 27
2025-06-16 05:00, 2025-06-16 05:41, 0, taxi, 41, walk, 00
2025-06-16 05:07, 2025-06-16 06:07, 0, walk, 10, taxi, 22
2025-06-16 05:15, 2025-06-16 06:23, 1, walk, 09, taxi, 27
2025-06-16 05:19, 2025-06-16 06:45, 1, walk, 10, walk, 03
2025-06-16 05:19, 2025-06-16 06:29, 0, walk, 10, taxi, 27
2025-06-16 05:39, 2025-06-16 06:45, 0, taxi, 17, walk, 03
2025-06-16 06:00, 2025-06-16 06:41, 0, taxi, 41, walk, 00
2025-06-16 06:29, 2025-06-16 09:16, 1, walk, 09, walk, 03
2025-06-16 06:29, 2025-06-16 07:37, 0, walk, 09, taxi, 27
2025-06-16 07:00, 2025-06-16 07:41, 0, taxi, 41, walk, 00
2025-06-16 08:00, 2025-06-16 08:41, 0, taxi, 41, walk, 00
2025-06-16 08:10, 2025-06-16 09:16, 0, taxi, 17, walk, 03
2025-06-16 08:19, 2025-06-16 10:43, 2, taxi, 05, walk, 03
2025-06-16 08:29, 2025-06-16 09:37, 0, walk, 09, taxi, 27
2025-06-16 08:57, 2025-06-16 09:52, 0, taxi, 27, walk, 03
2025-06-16 09:00, 2025-06-16 09:41, 0, taxi, 41, walk, 00
2025-06-16 09:25, 2025-06-16 10:23, 0, walk, 09, taxi, 27
2025-06-16 09:25, 2025-06-16 10:45, 1, walk, 09, walk, 03
2025-06-16 09:26, 2025-06-16 11:07, 1, walk, 09, taxi, 27
2025-06-16 09:29, 2025-06-16 10:31, 0, walk, 10, taxi, 37
2025-06-16 09:39, 2025-06-16 10:45, 0, taxi, 17, walk, 03
2025-06-16 10:00, 2025-06-16 10:41, 0, taxi, 41, walk, 00
2025-06-16 10:24, 2025-06-16 11:22, 0, walk, 09, taxi, 27
2025-06-16 10:28, 2025-06-16 11:37, 0, walk, 10, taxi, 27
2025-06-16 10:28, 2025-06-16 11:45, 1, walk, 10, walk, 03
2025-06-16 10:39, 2025-06-16 11:45, 0, taxi, 17, walk, 03
2025-06-16 11:00, 2025-06-16 11:41, 0, taxi, 41, walk, 00
2025-06-16 11:17, 2025-06-16 12:17, 0, walk, 10, taxi, 26
2025-06-16 11:28, 2025-06-16 12:45, 1, walk, 10, walk, 03
2025-06-16 11:34, 2025-06-16 12:32, 0, walk, 09, taxi, 27
2025-06-16 11:39, 2025-06-16 12:45, 0, taxi, 17, walk, 03
2025-06-16 12:00, 2025-06-16 12:41, 0, taxi, 41, walk, 00
2025-06-16 12:17, 2025-06-16 13:17, 0, walk, 10, taxi, 26
2025-06-16 12:24, 2025-06-16 13:45, 1, walk, 09, walk, 03
2025-06-16 12:24, 2025-06-16 13:22, 0, walk, 09, taxi, 27
2025-06-16 12:33, 2025-06-16 15:43, 3, walk, 10, walk, 03
2025-06-16 12:33, 2025-06-16 13:42, 0, walk, 10, taxi, 27
2025-06-16 12:39, 2025-06-16 13:45, 0, taxi, 17, walk, 03
2025-06-16 13:00, 2025-06-16 13:41, 0, taxi, 41, walk, 00
2025-06-16 13:03, 2025-06-16 15:43, 3, taxi, 27, walk, 03
2025-06-16 13:17, 2025-06-16 14:17, 0, walk, 10, taxi, 26
2025-06-16 13:19, 2025-06-16 15:43, 2, taxi, 05, walk, 03
2025-06-16 13:25, 2025-06-16 14:23, 0, walk, 09, taxi, 27
2025-06-16 13:28, 2025-06-16 14:37, 0, walk, 10, taxi, 27
2025-06-16 14:00, 2025-06-16 14:41, 0, taxi, 41, walk, 00
2025-06-16 14:29, 2025-06-16 15:47, 1, walk, 09, walk, 03
2025-06-16 14:29, 2025-06-16 15:37, 0, walk, 09, taxi, 27
2025-06-16 14:39, 2025-06-16 15:47, 0, taxi, 17, walk, 03
2025-06-16 15:00, 2025-06-16 15:41, 0, taxi, 41, walk, 00
2025-06-16 15:14, 2025-06-16 16:23, 1, walk, 09, taxi, 27
2025-06-16 15:29, 2025-06-16 16:37, 0, walk, 09, taxi, 27
2025-06-16 15:29, 2025-06-16 16:45, 1, walk, 09, walk, 03
2025-06-16 15:39, 2025-06-16 16:45, 0, taxi, 17, walk, 03
2025-06-16 16:00, 2025-06-16 16:41, 0, taxi, 41, walk, 00
2025-06-16 16:19, 2025-06-17 04:15, 2, taxi, 05, walk, 03
2025-06-16 16:27, 2025-06-17 04:15, 3, taxi, 27, walk, 03
2025-06-16 16:30, 2025-06-16 17:39, 0, walk, 10, taxi, 27
2025-06-16 17:00, 2025-06-16 17:41, 0, taxi, 41, walk, 00
2025-06-16 18:00, 2025-06-16 18:41, 0, taxi, 41, walk, 00
2025-06-16 19:00, 2025-06-16 19:41, 0, taxi, 41, walk, 00
2025-06-16 20:00, 2025-06-16 20:41, 0, taxi, 41, walk, 00
)__"sv;

constexpr auto const schleife_kleinpriebus_montag_out =
    R"__(departure, arrival, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
2025-06-16 02:00, 2025-06-16 02:41, 0, taxi, 41, walk, 00
2025-06-16 02:29, 2025-06-16 04:45, 1, walk, 09, walk, 03
2025-06-16 03:39, 2025-06-16 04:45, 0, taxi, 17, walk, 03
2025-06-16 05:19, 2025-06-16 06:45, 1, walk, 10, walk, 03
2025-06-16 06:29, 2025-06-16 09:16, 1, walk, 09, walk, 03
2025-06-16 09:25, 2025-06-16 10:45, 1, walk, 09, walk, 03
2025-06-16 10:28, 2025-06-16 11:45, 1, walk, 10, walk, 03
2025-06-16 11:28, 2025-06-16 12:45, 1, walk, 10, walk, 03
2025-06-16 12:24, 2025-06-16 13:45, 1, walk, 09, walk, 03
2025-06-16 12:33, 2025-06-16 15:43, 3, walk, 10, walk, 03
2025-06-16 14:29, 2025-06-16 15:47, 1, walk, 09, walk, 03
2025-06-16 15:29, 2025-06-16 16:45, 1, walk, 09, walk, 03
2025-06-16 17:00, 2025-06-16 17:41, 0, taxi, 41, walk, 00
2025-06-16 18:00, 2025-06-16 18:41, 0, taxi, 41, walk, 00
2025-06-16 19:00, 2025-06-16 19:41, 0, taxi, 41, walk, 00
2025-06-16 20:00, 2025-06-16 20:41, 0, taxi, 41, walk, 00
)__"sv;

TEST(odm, schleife_kleinpriebus_montag) {
  EXPECT_EQ(schleife_kleinpriebus_montag_out,
            mix(schleife_kleinpriebus_montag_in));
}