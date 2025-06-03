#include "gtest/gtest.h"

#include "motis/odm/mixer_reqs.h"

using namespace std::string_view_literals;
using namespace motis::odm;

constexpr auto const cs0 =
    R"__(departure_time, arrival_time, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration
12:34, 23:45, 3, taxi, 12, walk, 4
11:11, 22:22, 2, walk, 5, walk, 6
66:33, 77:44, 8, taxi, 20, taxi, 20
)__"sv;

TEST(odm_mixer_reqs, in_out) { EXPECT_EQ(cs0, to_csv(read(cs0))); }