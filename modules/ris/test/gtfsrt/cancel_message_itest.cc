#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"

#include "./gtfsrt_itest.h"

using namespace motis;
using namespace motis::test;
using namespace motis::module;
using namespace motis::ris::gtfsrt;

// used cancel message is the same as the one used in the
// cancel_message_test.cc
// test at t0 checks that trip can be found
// test at t1 checks that trip has deactivated lcons

constexpr auto const TIMEZONE_OFFSET = -7200;

struct ris_gtfsrt_cancel_message_itest_t0 : public gtfsrt_itest {
  ris_gtfsrt_cancel_message_itest_t0()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/cancel_itest/"
            "t0") {}
};

TEST_F(ris_gtfsrt_cancel_message_itest_t0, before_cancel) {
  auto const trip =
      get_trip(sched(), "8503000:0:41/42", 0, 1561597200 + TIMEZONE_OFFSET,
               "8502113:0:4", 1561600800 + TIMEZONE_OFFSET, "1");

  ASSERT_NE(nullptr, trip);
  auto const lcon_idx = trip->lcon_idx_;
  for (auto const& e : *trip->edges_) {
    EXPECT_TRUE(e->m_.route_edge_.conns_[lcon_idx].valid_);
  }
}

struct ris_gtfsrt_cancel_message_itest_t1 : public gtfsrt_itest {
  ris_gtfsrt_cancel_message_itest_t1()
      : gtfsrt_itest(
            "--ris.input=modules/ris/test_resources/gtfs-rt/cancel_itest/"
            "t1") {}
};

TEST_F(ris_gtfsrt_cancel_message_itest_t1, after_cancel) {
  auto const trip =
      get_trip(sched(), "8503000:0:41/42", 0, 1561597200 + TIMEZONE_OFFSET,
               "8502113:0:4", 1561600800 + TIMEZONE_OFFSET, "1");

  ASSERT_NE(nullptr, trip);
  auto const lcon_idx = trip->lcon_idx_;
  for (auto const& e : *trip->edges_) {
    EXPECT_FALSE(e->m_.route_edge_.conns_[lcon_idx].valid_);
  }
}