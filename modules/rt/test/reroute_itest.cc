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

struct rt_reroute_test : public motis_instance_test {
  rt_reroute_test()
      : motis::test::motis_instance_test(
            no_rule_services(dataset_opt), {"ris", "rt"},
            {"--ris.input=test/schedule/invalid_realtime/risml/reroute.xml",
             "--ris.init_time=2015-11-24T10:10:00"}) {}

  static loader::loader_options no_rule_services(loader::loader_options opt) {
    opt.apply_rules_ = false;
    return opt;
  }
};

TEST_F(rt_reroute_test, reroute_with_delay_times) {
  auto evs = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(motis_time(910), evs.at("0000005").dep_);
  EXPECT_EQ(motis_time(1105), evs.at("0000002").arr_);
  EXPECT_EQ(motis_time(1112), evs.at("0000002").dep_);
  EXPECT_EQ(motis_time(1305), evs.at("0000004").arr_);
  EXPECT_EQ(motis_time(1312), evs.at("0000004").dep_);
  EXPECT_EQ(motis_time(1500), evs.at("0000001").arr_);
}

TEST_F(rt_reroute_test, reroute_with_delay_in_out) {
  auto ev1 = get_trip_event_info(
      sched(), get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381"));
  EXPECT_EQ(in_out_allowed(true, true), ev1.at("0000005").in_out_);
  EXPECT_EQ(in_out_allowed(true, true), ev1.at("0000002").in_out_);
  EXPECT_EQ(in_out_allowed(false, false), ev1.at("0000004").in_out_);
  EXPECT_EQ(in_out_allowed(true, true), ev1.at("0000001").in_out_);
}
