#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"
#include "motis/rt/separate_trip.h"
#include "motis/rt/update_msg_builder.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/invalid_realtime.h"

using namespace motis;
using namespace motis::rt;
using namespace motis::test;
using motis::test::schedule::invalid_realtime::dataset_opt;

struct rt_trip_separation_test : public motis_instance_test {
  rt_trip_separation_test()
      : motis::test::motis_instance_test(dataset_opt, {"rt"}) {}
};

TEST_F(rt_trip_separation_test, simple) {
  auto t1_d1 = get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                        unix_time(1400), "381");
  auto t1_d2 = get_trip(sched(), "0000001", 1, unix_time(1010, 1), "0000005",
                        unix_time(1400, 1), "381");
  auto t2_d1 = get_trip(sched(), "0000001", 2, unix_time(1010), "0000004",
                        unix_time(1300), "382");
  auto t2_d2 = get_trip(sched(), "0000001", 2, unix_time(1010, 1), "0000004",
                        unix_time(1300, 1), "382");

  EXPECT_EQ(t1_d1->edges_->at(0).get_edge(), t1_d2->edges_->at(0).get_edge());
  EXPECT_EQ(t2_d1->edges_->at(0).get_edge(), t2_d2->edges_->at(0).get_edge());

  auto const sched_res_id = to_res_id(motis::module::global_res_id::SCHEDULE);
  auto& sched = *instance_->get<schedule_data>(sched_res_id).schedule_;
  auto update_builder = update_msg_builder{sched, sched_res_id};
  separate_trip(sched,
                ev_key{t1_d1->edges_->at(0).get_edge(), t1_d1->lcon_idx_,
                       event_type::DEP},
                update_builder);

  EXPECT_NE(t1_d1->edges_->at(0).get_edge(), t1_d2->edges_->at(0).get_edge());
  EXPECT_NE(t2_d1->edges_->at(0).get_edge(), t2_d2->edges_->at(0).get_edge());
}
