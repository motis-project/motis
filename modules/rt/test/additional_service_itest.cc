#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/ris/risml/risml_parser.h"
#include "motis/rt/separate_trip.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/invalid_realtime.h"

using namespace motis;
using namespace motis::rt;
using namespace motis::test;
using namespace motis::module;
using motis::test::schedule::invalid_realtime::dataset_opt;

struct rt_additional_service_test : public motis_instance_test {
  rt_additional_service_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"ris", "rt"},
            {"--ris.input=test/schedule/invalid_realtime/risml/additional.xml",
             "--ris.init_time=2015-11-24T22:00:00"}) {}
};

mcd::vector<mcd::string> get_tracks(motis::schedule const& sched,
                                    trip const* trp) {
  mcd::vector<mcd::string> tracks;
  for (auto const& trp_e : *trp->edges_) {
    auto const full_con = trp_e.get_edge()->m_.route_edge_.conns_[0].full_con_;
    tracks.push_back(sched.tracks_[full_con->d_track_]);
    tracks.push_back(sched.tracks_[full_con->a_track_]);
  }
  return tracks;
}

std::vector<motis::time> get_times(trip const* trp) {
  std::vector<motis::time> times;
  for (auto const& trp_e : *trp->edges_) {
    auto const lcon = trp_e.get_edge()->m_.route_edge_.conns_[0];
    times.push_back(lcon.d_time_);
    times.push_back(lcon.a_time_);
  }
  return times;
}

TEST_F(rt_additional_service_test, simple) {
  auto trp = get_trip(sched(), "0000001", 77, unix_time(2200), "0000004",
                      unix_time(2300), "");

  for (auto const& trp_e : *trp->edges_) {
    auto const& conns = trp_e.get_edge()->m_.route_edge_.conns_;
    ASSERT_EQ(1, conns.size());

    auto const& trps = *sched().merged_trips_.at(conns[0].trips_);
    ASSERT_EQ(1, trps.size());
    EXPECT_EQ(trp, trps[0]);

    EXPECT_EQ(77, conns[0].full_con_->con_info_->train_nr_);
    EXPECT_EQ(
        "ICE",
        sched().categories_[conns[0].full_con_->con_info_->family_]->name_);
  }

  EXPECT_EQ(mcd::vector<mcd::string>({"7a", "2", "2", "5"}),
            get_tracks(sched(), trp));

  EXPECT_EQ(std::vector<motis::time>({motis_time(2200), motis_time(2225),
                                      motis_time(2230), motis_time(2300)}),
            get_times(trp));
}
