#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"

#include "./gtfsrt_itest.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::ris::gtfsrt;

// used delay messages are the same as the ones used in the
// delay_message_test.cc
// test t0 is some time after a delay for a trip was received
// test t1 is some time after a second delay for the same trip was received
// the second update should be applied independend of the second

constexpr auto const TIMEZONE_OFFSET = -120;

struct ris_gtfsrt_delay_message_itest_t0 : public gtfsrt_itest {
  ris_gtfsrt_delay_message_itest_t0()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/delay_itest/t0") {}
};

TEST_F(ris_gtfsrt_delay_message_itest_t0, simple) {
  auto trp = get_trip(sched(), "8502113", 0, 1561597620 + TIMEZONE_OFFSET * 60,
                      "8500309", 1561598940 + TIMEZONE_OFFSET * 60, "1");

  auto evs = get_trip_event_info(sched(), trp);
  EXPECT_EQ(motis_time(1561597620) + TIMEZONE_OFFSET, evs.at("8502113").dep_);
  EXPECT_EQ(motis_time(1561597860) + TIMEZONE_OFFSET, evs.at("8502114").arr_);
  EXPECT_EQ(motis_time(1561597860) + TIMEZONE_OFFSET, evs.at("8502114").dep_);
  EXPECT_EQ(motis_time(1561598100) + TIMEZONE_OFFSET, evs.at("8502119").arr_);
  EXPECT_EQ(motis_time(1561598160) + TIMEZONE_OFFSET, evs.at("8502119").dep_);
  EXPECT_EQ(motis_time(1561598340) + TIMEZONE_OFFSET, evs.at("8502105").arr_);
  EXPECT_EQ(motis_time(1561598340) + TIMEZONE_OFFSET, evs.at("8502105").dep_);
  EXPECT_EQ(motis_time(1561598460) + TIMEZONE_OFFSET, evs.at("8502247").arr_);
  EXPECT_EQ(motis_time(1561598520) + TIMEZONE_OFFSET, evs.at("8502247").dep_);
  EXPECT_EQ(motis_time(1561598640) + TIMEZONE_OFFSET, evs.at("8502237").arr_);
  EXPECT_EQ(motis_time(1561598640) + TIMEZONE_OFFSET, evs.at("8502237").dep_);
  EXPECT_EQ(motis_time(1561598700) + TIMEZONE_OFFSET, evs.at("8500309").arr_);
}

struct ris_gtfsrt_delay_message_itest_t1 : public gtfsrt_itest {
  ris_gtfsrt_delay_message_itest_t1()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/delay_itest/t1") {}
};

TEST_F(ris_gtfsrt_delay_message_itest_t1, updated_delay) {
  auto trp = get_trip(sched(), "8502113", 0, 1561597620 + TIMEZONE_OFFSET * 60,
                      "8500309", 1561598940 + TIMEZONE_OFFSET * 60, "1");

  auto evs = get_trip_event_info(sched(), trp);
  EXPECT_EQ(motis_time(1561597620) + TIMEZONE_OFFSET, evs.at("8502113").dep_);
  EXPECT_EQ(motis_time(1561597860) + TIMEZONE_OFFSET, evs.at("8502114").arr_);
  EXPECT_EQ(motis_time(1561597860) + TIMEZONE_OFFSET, evs.at("8502114").dep_);
  EXPECT_EQ(motis_time(1561598100) + TIMEZONE_OFFSET, evs.at("8502119").arr_);
  EXPECT_EQ(motis_time(1561598160) + TIMEZONE_OFFSET, evs.at("8502119").dep_);
  EXPECT_EQ(motis_time(1561598340) + TIMEZONE_OFFSET, evs.at("8502105").arr_);
  EXPECT_EQ(motis_time(1561598340) + TIMEZONE_OFFSET, evs.at("8502105").dep_);
  EXPECT_EQ(motis_time(1561598460) + TIMEZONE_OFFSET, evs.at("8502247").arr_);
  EXPECT_EQ(motis_time(1561598700) + TIMEZONE_OFFSET, evs.at("8502247").dep_);
  EXPECT_EQ(motis_time(1561598820) + TIMEZONE_OFFSET, evs.at("8502237").arr_);
  EXPECT_EQ(motis_time(1561598820) + TIMEZONE_OFFSET, evs.at("8502237").dep_);
  EXPECT_EQ(motis_time(1561599240) + TIMEZONE_OFFSET, evs.at("8500309").arr_);
}