#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"

#include "./gtfsrt_itest.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::ris::gtfsrt;

// used additional message is the same as the one used in the
// additional_message_test.cc
// Two situations are to be tested here
//  1. at t0 a additional train is added to the schedule
//	   at t1 the same train is reported again but with delay
//
//  2. at t0 a additional train is added which already contains delays and
//			stop skips
//     at t1 a different delay and further skips for this train are reported

struct ris_gtfsrt_addition_message_itest_t0 : public gtfsrt_itest {
  ris_gtfsrt_addition_message_itest_t0()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/addition_itest/"
            "t0") {}
};

TEST_F(ris_gtfsrt_addition_message_itest_t0,
       check_initial_addition_train_handling_at_t0) {
  auto trp_added_delay = get_trip(sched(), "8501026:0:2", 0, 1561597920,
                                  "8503000:0:14", 1561600800, "9-1-A-j19-1");

  EXPECT_EQ(2, trp_added_delay->edges_->size());
  auto trip_events = get_trip_event_info(sched(), trp_added_delay);
  EXPECT_EQ(3, trip_events.size());

  EXPECT_EQ(motis_time(1561597920),
            trip_events.at("8501026:0:2").dep_);  // seq 1
  EXPECT_EQ(motis_time(1561599120),
            trip_events.at("8501008:0:2").arr_);  // seq 2
  EXPECT_EQ(motis_time(1561599300),
            trip_events.at("8501008:0:2").dep_);  // seq 2
  EXPECT_EQ(motis_time(1561600740),
            trip_events.at("8503000:0:14").arr_);  // seq 3

  auto trp_added_skip = get_trip(sched(), "8501008:0:2", 0, 1561597920,
                                 "8501500:0:3", 1561602000, "9-1-A-j19-2");

  EXPECT_EQ(2, trp_added_skip->edges_->size());
  trip_events = get_trip_event_info(sched(), trp_added_skip);
  EXPECT_EQ(3, trip_events.size());

  EXPECT_EQ(motis_time(1561597920),
            trip_events.at("8501008:0:2").dep_);  // seq 1
  EXPECT_EQ(motis_time(1561599120),
            trip_events.at("8501026:0:2").arr_);  // seq 2
  EXPECT_EQ(motis_time(1561599180),
            trip_events.at("8501026:0:2").dep_);  // seq 2
  // EXPECT_EQ(motis_time(1561601880), trip_events.at("8501500").arr_);  // seq
  // 4
}

struct ris_gtfsrt_addition_message_itest_t1 : public gtfsrt_itest {
  ris_gtfsrt_addition_message_itest_t1()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/addition_itest/"
            "t1") {}
};

TEST_F(ris_gtfsrt_addition_message_itest_t1,
       check_addition_train_handling_at_t1) {
  auto trp_added_delay = get_trip(sched(), "8501026:0:2", 0, 1561597920,
                                  "8503000:0:14", 1561600800, "9-1-A-j19-1");

  EXPECT_EQ(2, trp_added_delay->edges_->size());
  auto trip_events = get_trip_event_info(sched(), trp_added_delay);
  EXPECT_EQ(3, trip_events.size());

  EXPECT_EQ(motis_time(1561597920),
            trip_events.at("8501026:0:2").dep_);  // seq 1
  EXPECT_EQ(motis_time(1561599240),
            trip_events.at("8501008:0:2").arr_);  // seq 2
  // EXPECT_EQ(motis_time(1561599420), trip_events.at("8501008").dep_);  // seq
  // 2 - propagated time
  EXPECT_EQ(motis_time(1561600740),
            trip_events.at("8503000:0:14").arr_);  // seq 3

  auto trp_added_skip = get_trip(sched(), "8501008:0:2", 0, 1561597920,
                                 "8501500:0:3", 1561602000, "9-1-A-j19-2");

  EXPECT_EQ(1, trp_added_skip->edges_->size());
  trip_events = get_trip_event_info(sched(), trp_added_skip);
  EXPECT_EQ(2, trip_events.size());

  EXPECT_EQ(motis_time(1561597920),
            trip_events.at("8501008:0:2").dep_);  // seq 1
  EXPECT_EQ(motis_time(1561599120),
            trip_events.at("8501026:0:2").arr_);  // seq 2
}