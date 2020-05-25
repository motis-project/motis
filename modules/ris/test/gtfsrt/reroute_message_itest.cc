#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"

#include "motis/ris/gtfs-rt/common.h"

#include "./gtfsrt_itest.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::ris::gtfsrt;

constexpr auto const TIMEZONE_OFFSET = -120;

// used reroute message is a combination of the one used in the
// delay_message_test.cc and further ones listed below Two Situations are
// checked:
//   1. First stop skips for a trip are received and processed.
//	Second the same skips for the same trip are received and processed.
//
//   2. First stop skips for a trip are received and processed.
//	Second the same and further skips are received and processed.
//
// test at t0 checks that skips are parsed correctly
// test at t1 checks that no further skips are added by the same message
struct ris_gtfsrt_reroute_message_itest_t0 : public gtfsrt_itest {
  ris_gtfsrt_reroute_message_itest_t0()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/reroute_itest/"
            "/t0") {}
};

TEST_F(ris_gtfsrt_reroute_message_itest_t0,
       handle_initial_reported_skips_and_delay) {
  auto trp_same_skips =
      get_trip(sched(), "95.TA.59-100-j19-1.91.H", 1561593600);

  EXPECT_EQ(3, trp_same_skips->edges_->size());
  auto trip_events = get_trip_event_info(sched(), trp_same_skips);

  EXPECT_EQ(4, trip_events.size());
  EXPECT_EQ(motis_time(1561612200) + TIMEZONE_OFFSET,
            trip_events.at("8501300:0:3").dep_);  // seq 2
  EXPECT_EQ(motis_time(1561612800) + TIMEZONE_OFFSET,
            trip_events.at("8501400:0:1").arr_);  // seq 3
  EXPECT_EQ(motis_time(1561612860) + TIMEZONE_OFFSET,
            trip_events.at("8501400:0:1").dep_);  // seq 3
  EXPECT_EQ(motis_time(1561613220) + TIMEZONE_OFFSET,
            trip_events.at("8501402:0:1").arr_);  // seq 4
  EXPECT_EQ(motis_time(1561613280) + TIMEZONE_OFFSET,
            trip_events.at("8501402:0:1").dep_);  // seq 4
  EXPECT_EQ(motis_time(1561614180) + TIMEZONE_OFFSET,
            trip_events.at("8501500:0:2").arr_);  // seq 6

  auto trp_skip_delay = get_trip(sched(), "13.TA.1-1-j19-1.13.H", 1561593600);

  EXPECT_EQ(7, trp_skip_delay->edges_->size());
  trip_events = get_trip_event_info(sched(), trp_skip_delay);

  EXPECT_EQ(8, trip_events.size());
  EXPECT_EQ(motis_time(1561614000) + TIMEZONE_OFFSET,
            trip_events.at("8502119:0:3").dep_);  // seq 2
  EXPECT_EQ(motis_time(1561614360) + TIMEZONE_OFFSET,
            trip_events.at("8502113:0:4").arr_);  // seq 3
  EXPECT_EQ(motis_time(1561614480) + TIMEZONE_OFFSET,
            trip_events.at("8502113:0:4").dep_);  // seq 3
  EXPECT_EQ(motis_time(1561615020) + TIMEZONE_OFFSET,
            trip_events.at("8500218:0:8").arr_);  // seq 4
  EXPECT_EQ(motis_time(1561615440) + TIMEZONE_OFFSET,
            trip_events.at("8500218:0:8").dep_);  // seq 4
  EXPECT_EQ(motis_time(1561617000) + TIMEZONE_OFFSET,
            trip_events.at("8507000:0:5").arr_);  // seq 6
}

struct ris_gtfsrt_reroute_message_itest_t1 : public gtfsrt_itest {
  ris_gtfsrt_reroute_message_itest_t1()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/reroute_itest/"
            "/t1") {}
};

TEST_F(ris_gtfsrt_reroute_message_itest_t1,
       handle_further_reported_skips_and_delay) {
  auto trp_same_skips =
      get_trip(sched(), "95.TA.59-100-j19-1.91.H", 1561593600);
  // expecting same state as with t0
  EXPECT_EQ(3, trp_same_skips->edges_->size());
  auto trip_events = get_trip_event_info(sched(), trp_same_skips);

  EXPECT_EQ(4, trip_events.size());
  EXPECT_EQ(motis_time(1561612200) + TIMEZONE_OFFSET,
            trip_events.at("8501300:0:3").dep_);  // seq 2
  EXPECT_EQ(motis_time(1561612800) + TIMEZONE_OFFSET,
            trip_events.at("8501400:0:1").arr_);  // seq 3
  EXPECT_EQ(motis_time(1561612860) + TIMEZONE_OFFSET,
            trip_events.at("8501400:0:1").dep_);  // seq 3
  EXPECT_EQ(motis_time(1561613220) + TIMEZONE_OFFSET,
            trip_events.at("8501402:0:1").arr_);  // seq 4
  EXPECT_EQ(motis_time(1561613280) + TIMEZONE_OFFSET,
            trip_events.at("8501402:0:1").dep_);  // seq 4
  EXPECT_EQ(motis_time(1561614180) + TIMEZONE_OFFSET,
            trip_events.at("8501500:0:2").arr_);  // seq 6

  // expecting to also skip seq-no 6
  auto trp_skip_delay = get_trip(sched(), "13.TA.1-1-j19-1.13.H", 1561593600);

  EXPECT_EQ(6, trp_skip_delay->edges_->size());
  trip_events = get_trip_event_info(sched(), trp_skip_delay);

  EXPECT_EQ(7, trip_events.size());
  EXPECT_EQ(motis_time(1561614000) + TIMEZONE_OFFSET,
            trip_events.at("8502119:0:3").dep_);  // seq 2
  EXPECT_EQ(motis_time(1561614360) + TIMEZONE_OFFSET,
            trip_events.at("8502113:0:4").arr_);  // seq 3
  EXPECT_EQ(motis_time(1561614480) + TIMEZONE_OFFSET,
            trip_events.at("8502113:0:4").dep_);  // seq 3
  EXPECT_EQ(motis_time(1561615020) + TIMEZONE_OFFSET,
            trip_events.at("8500218:0:8").arr_);  // seq 4
}