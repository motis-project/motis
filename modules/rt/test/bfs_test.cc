#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"

#include "motis/rt/separate_trip.h"

#include "motis/test/motis_instance_test.h"

using namespace motis;
using namespace motis::rt;
using namespace motis::test;
using namespace motis::module;

struct bfs_test : public motis_instance_test {
  bfs_test()
      : motis::test::motis_instance_test(
            loader::loader_options{
                {"base/loader/test_resources/hrd_schedules/mss-ts"},
                "20150329"},
            {"rt"}) {}
};

TEST_F(bfs_test, simple) {
  auto trp = get_trip(sched(), "0000001", 1, unix_time(110, 0, 60), "0000007",
                      unix_time(600, 0, 120), "");

  auto first_dep =
      ev_key{trp->edges_->front().get_edge(), trp->lcon_idx_, event_type::DEP};

  std::set<trip::route_edge> trp_edges;
  for (auto const& t : route_trips(sched(), first_dep)) {
    trp_edges.insert(begin(*t->edges_), end(*t->edges_));
  }

  auto bfs_edges = route_bfs(first_dep, bfs_direction::BOTH, false);

  EXPECT_EQ(trp_edges, bfs_edges);
}
