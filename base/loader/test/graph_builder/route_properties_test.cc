#include "gtest/gtest.h"

#include <cinttypes>
#include <climits>

#include "motis/core/common/date_time_util.h"
#include "motis/core/schedule/time.h"

#include "./graph_builder_test.h"

namespace motis::loader {

class loader_graph_builder_never_meet : public loader_graph_builder_test {
public:
  loader_graph_builder_never_meet()
      : loader_graph_builder_test("never-meet", "20150104", 7) {}
};

TEST_F(loader_graph_builder_never_meet, routes) {
  ASSERT_EQ(3, sched_->route_index_to_first_route_node_.size());

  auto next_search_time = time{};

  {
    auto const node_it = begin(sched_->route_index_to_first_route_node_);
    auto const connections = get_connections(*node_it, time{0});

    ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
    EXPECT_EQ(2, connections.size());
    EXPECT_EQ(1, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);

    auto const [lcon, day, dep_node, arr_node] = connections[0];
    next_search_time = lcon->event_time(event_type::DEP, day);
  }

  {
    auto const node_it = begin(sched_->route_index_to_first_route_node_);
    auto const connections = get_connections(*node_it, next_search_time + 1);

    auto const [lcon, day, dep_node, arr_node] = connections[0];
    EXPECT_EQ(2, connections.size());
    EXPECT_EQ(4, lcon->full_con_->con_info_->train_nr_);
  }
  {
    auto const node_it =
        std::next(begin(sched_->route_index_to_first_route_node_), 1);
    auto const connections = get_connections(*node_it, time{0});

    ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
    EXPECT_EQ(2, connections.size());
    EXPECT_EQ(2, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);
  }
  {
    auto const node_it =
        std::next(begin(sched_->route_index_to_first_route_node_), 2);
    auto const connections = get_connections(*node_it, time{0});
    ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
    EXPECT_EQ(3, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);
  }
}

}  // namespace motis::loader
