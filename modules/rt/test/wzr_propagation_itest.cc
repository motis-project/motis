#include "gtest/gtest.h"

#include <iostream>
#include <map>

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/wzr_realtime.h"

#include "./get_trip_event_info.h"

using namespace motis;
using namespace motis::module;
using namespace motis::rt;
using namespace motis::test;
using namespace motis::test::schedule;
using motis::test::schedule::wzr_realtime::dataset_opt;

struct rt_wzr_propagation1_test : public motis_instance_test {
  rt_wzr_propagation1_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/wzr_realtime/risml/"
             "delays1.xml",
             "--ris.init_time=2015-11-24T10:00:00"}) {}
};

TEST_F(rt_wzr_propagation1_test, wzr_propagation_test) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(1010), ev1["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev1["0000002"].arr_);
  EXPECT_EQ(motis_time(1301), ev1["0000002"].dep_);
  EXPECT_EQ(motis_time(1351), ev1["0000003"].arr_);
  EXPECT_EQ(motis_time(1353), ev1["0000003"].dep_);
  EXPECT_EQ(motis_time(1443), ev1["0000004"].arr_);
  EXPECT_EQ(motis_time(1445), ev1["0000004"].dep_);
  EXPECT_EQ(motis_time(1535), ev1["0000005"].arr_);

  auto ev2 = get_trip_event_info(
      sched(), get_trip(sched(), "0000006", 2, unix_time(1025), "0000009",
                        unix_time(1420), "382"));
  EXPECT_EQ(motis_time(1025), ev2["0000006"].dep_);
  EXPECT_EQ(motis_time(1120), ev2["0000007"].arr_);
  EXPECT_EQ(motis_time(1125), ev2["0000007"].dep_);
  EXPECT_EQ(motis_time(1220), ev2["0000003"].arr_);
  EXPECT_EQ(motis_time(1225), ev2["0000003"].dep_);
  EXPECT_EQ(motis_time(1320), ev2["0000008"].arr_);
  EXPECT_EQ(motis_time(1325), ev2["0000008"].dep_);
  EXPECT_EQ(motis_time(1420), ev2["0000009"].arr_);
}

struct rt_wzr_propagation2_test : public motis_instance_test {
  rt_wzr_propagation2_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/wzr_realtime/risml/"
             "delays2.xml",
             "--ris.init_time=2015-11-24T10:01:00"}) {}
};

TEST_F(rt_wzr_propagation2_test, wzr_propagation_test) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(1010), ev1["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev1["0000002"].arr_);
  EXPECT_EQ(motis_time(1137), ev1["0000002"].dep_);
  EXPECT_EQ(motis_time(1227), ev1["0000003"].arr_);
  EXPECT_EQ(motis_time(1229), ev1["0000003"].dep_);
  EXPECT_EQ(motis_time(1319), ev1["0000004"].arr_);
  EXPECT_EQ(motis_time(1321), ev1["0000004"].dep_);
  EXPECT_EQ(motis_time(1411), ev1["0000005"].arr_);

  auto ev2 = get_trip_event_info(
      sched(), get_trip(sched(), "0000006", 2, unix_time(1025), "0000009",
                        unix_time(1420), "382"));
  EXPECT_EQ(motis_time(1025), ev2["0000006"].dep_);
  EXPECT_EQ(motis_time(1120), ev2["0000007"].arr_);
  EXPECT_EQ(motis_time(1125), ev2["0000007"].dep_);
  EXPECT_EQ(motis_time(1220), ev2["0000003"].arr_);
  EXPECT_EQ(motis_time(1229), ev2["0000003"].dep_);
  EXPECT_EQ(motis_time(1324), ev2["0000008"].arr_);
  EXPECT_EQ(motis_time(1326), ev2["0000008"].dep_);
  EXPECT_EQ(motis_time(1421), ev2["0000009"].arr_);
}

struct rt_wzr_propagation3_test : public motis_instance_test {
  rt_wzr_propagation3_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/wzr_realtime/risml/",
             "--ris.init_time=2015-11-24T10:02:00"}) {}
};

TEST_F(rt_wzr_propagation3_test, wzr_propagation_test) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(1010), ev1["0000001"].dep_);
  EXPECT_EQ(motis_time(1100), ev1["0000002"].arr_);

  auto ev2 = get_trip_event_info(
      sched(), get_trip(sched(), "0000006", 2, unix_time(1025), "0000009",
                        unix_time(1420), "382"));
  EXPECT_EQ(motis_time(1025), ev2["0000006"].dep_);
  EXPECT_EQ(motis_time(1120), ev2["0000007"].arr_);
  EXPECT_EQ(motis_time(1125), ev2["0000007"].dep_);
  EXPECT_EQ(motis_time(1220), ev2["0000003"].arr_);
  EXPECT_EQ(motis_time(1225), ev2["0000003"].dep_);
  EXPECT_EQ(motis_time(1320), ev2["0000008"].arr_);
  EXPECT_EQ(motis_time(1325), ev2["0000008"].dep_);
  EXPECT_EQ(motis_time(1420), ev2["0000009"].arr_);
}
