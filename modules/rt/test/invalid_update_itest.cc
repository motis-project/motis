#include "gtest/gtest.h"

#include <map>

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/invalid_realtime.h"

#include "./get_trip_event_info.h"

using namespace motis;
using namespace motis::module;
using namespace motis::rt;
using namespace motis::test;
using namespace motis::test::schedule;
using motis::test::schedule::invalid_realtime::dataset_opt;

struct rt_invalid_trip_update_test : public motis_instance_test {
  rt_invalid_trip_update_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/invalid_realtime/risml/"
             "trip_conflict.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}
};

TEST_F(rt_invalid_trip_update_test, trip_conflict_test) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(1010), ev1["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev1["0000002"].arr_);
  EXPECT_EQ(motis_time(1301), ev1["0000002"].dep_);
  EXPECT_EQ(motis_time(1301), ev1["0000003"].arr_);
  EXPECT_EQ(motis_time(1301), ev1["0000003"].dep_);
  EXPECT_EQ(motis_time(1301), ev1["0000004"].arr_);
  EXPECT_EQ(motis_time(1310), ev1["0000004"].dep_);
  EXPECT_EQ(motis_time(1400), ev1["0000005"].arr_);

  auto ev2 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 2, unix_time(1010), "0000004",
                        unix_time(1300), "382"));
  EXPECT_EQ(motis_time(1010), ev2["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev2["0000002"].arr_);
  EXPECT_EQ(motis_time(1110), ev2["0000002"].dep_);
  EXPECT_EQ(motis_time(1200), ev2["0000003"].arr_);
  EXPECT_EQ(motis_time(1210), ev2["0000003"].dep_);
  EXPECT_EQ(motis_time(1300), ev2["0000004"].arr_);
}

struct rt_invalid_ts_update_test : public motis_instance_test {
  rt_invalid_ts_update_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/invalid_realtime/risml/ts_conflict.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}
};

TEST_F(rt_invalid_ts_update_test, ts_conflict_test) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(1010), ev1["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev1["0000002"].arr_);
  EXPECT_EQ(motis_time(1110), ev1["0000002"].dep_);
  EXPECT_EQ(motis_time(1200), ev1["0000003"].arr_);
  EXPECT_EQ(motis_time(1200), ev1["0000003"].dep_);
  EXPECT_EQ(motis_time(1200), ev1["0000004"].arr_);
  EXPECT_EQ(motis_time(1200), ev1["0000004"].dep_);
  EXPECT_EQ(motis_time(1400), ev1["0000005"].arr_);

  auto ev2 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 2, unix_time(1010), "0000004",
                        unix_time(1300), "382"));
  EXPECT_EQ(motis_time(1010), ev2["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev2["0000002"].arr_);
  EXPECT_EQ(motis_time(1110), ev2["0000002"].dep_);
  EXPECT_EQ(motis_time(1200), ev2["0000003"].arr_);
  EXPECT_EQ(motis_time(1200), ev2["0000003"].dep_);
  EXPECT_EQ(motis_time(1200), ev2["0000004"].arr_);
}
