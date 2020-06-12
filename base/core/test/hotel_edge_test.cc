#include "gtest/gtest.h"

#include "motis/core/schedule/edges.h"
#include "motis/core/access/station_access.h"

#include "motis/test/motis_instance_test.h"

namespace motis {

TEST(core_late_connections, time_off) {
  ASSERT_EQ(100, edge::calc_time_off(100, 200, 0));
  ASSERT_EQ(0, edge::calc_time_off(100, 200, 100));
  ASSERT_EQ(0, edge::calc_time_off(100, 200, 150));
  ASSERT_EQ(0, edge::calc_time_off(100, 200, 200));
  ASSERT_EQ(1240, edge::calc_time_off(100, 200, 300));
  ASSERT_EQ(101, edge::calc_time_off(100, 200, 1439));

  ASSERT_EQ(0, edge::calc_time_off(1340, 40, 0));
  ASSERT_EQ(0, edge::calc_time_off(1340, 40, 40));
  ASSERT_EQ(1200, edge::calc_time_off(1340, 40, 140));
  ASSERT_EQ(1, edge::calc_time_off(1340, 40, 1339));
  ASSERT_EQ(0, edge::calc_time_off(1340, 40, 1340));
  ASSERT_EQ(0, edge::calc_time_off(1340, 40, 1439));
}

class core_hotel_edges : public test::motis_instance_test {
protected:
  core_hotel_edges()
      : test::motis_instance_test({{"test/schedule/schedule_hotels/"},
                                   "20151019",
                                   2,
                                   false,
                                   false,
                                   false,
                                   true},
                                  {""}) {}
};

TEST_F(core_hotel_edges, test_hotels_edges) {
  std::vector<edge> hotel_edges = {
      make_hotel_edge(get_station_node(sched(), "1111111"), 420, 360, 5000, 3),
      make_hotel_edge(get_station_node(sched(), "2222222"), 480, 540, 4000, 3)};

  /* checkout-time: 7 * 60, min-stay-duration: 6 * 60 */
  {
    auto const& he = hotel_edges[0];
    auto const cost = he.get_edge_cost(23 * 60 + 59, nullptr);
    ASSERT_FALSE(cost.transfer_);
    ASSERT_EQ(5000, cost.price_);
    ASSERT_EQ(7 * 60 + 1, cost.time_);
    ASSERT_EQ(6 * 60 + 10, he.get_edge_cost(1440 + 50, nullptr).time_);
    ASSERT_EQ(6 * 60, he.get_edge_cost(1440 + 70, nullptr).time_);
    ASSERT_EQ(6 * 60, he.get_edge_cost(6 * 59, nullptr).time_);
    ASSERT_EQ(1440, he.get_edge_cost(7 * 60, nullptr).time_);

    ASSERT_EQ(5000, he.get_minimum_cost().price_);
    ASSERT_EQ(0, he.get_minimum_cost().time_);
    ASSERT_FALSE(he.get_minimum_cost().transfer_);
  }

  /* checkout-time: 8 * 60, min-stay-duration: 9 * 60 */
  {
    auto const& he = hotel_edges[1];
    auto const cost = he.get_edge_cost(22 * 60 + 59, nullptr);
    ASSERT_FALSE(cost.transfer_);
    ASSERT_EQ(4000, cost.price_);
    ASSERT_EQ(9 * 60 + 1, cost.time_);
    ASSERT_EQ(9 * 60, he.get_edge_cost(23 * 60 + 1, nullptr).time_);
  }
}

}  // namespace motis
