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

  auto node_it = begin(sched_->route_index_to_first_route_node_);
  auto connections = get_connections(*node_it, 0);

  ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
  EXPECT_EQ(2, connections.size());
  EXPECT_EQ(1, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);

  connections =
      get_connections(*node_it, std::get<0>(connections[0])->d_time_ + 1);
  EXPECT_EQ(2, connections.size());
  EXPECT_EQ(4, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);

  node_it = std::next(node_it, 1);
  connections = get_connections(*node_it, 0);
  ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
  EXPECT_EQ(2, connections.size());
  EXPECT_EQ(2, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);

  node_it = std::next(node_it, 1);
  connections = get_connections(*node_it, 0);
  ASSERT_TRUE(node_it != end(sched_->route_index_to_first_route_node_));
  EXPECT_EQ(3, std::get<0>(connections[0])->full_con_->con_info_->train_nr_);
}

class loader_graph_builder_duplicates_check : public loader_graph_builder_test {
public:
  loader_graph_builder_duplicates_check()
      : loader_graph_builder_test("duplicates", "20150104", 7) {}

  uint32_t get_train_num(char const* first_stop_id) {
    auto it = std::find_if(
        begin(sched_->route_index_to_first_route_node_),
        end(sched_->route_index_to_first_route_node_), [&](node const* n) {
          return sched_->stations_[n->get_station()->id_]->eva_nr_ ==
                 first_stop_id;
        });
    return std::get<0>(get_connections(*it, 0).at(0))
        ->full_con_->con_info_->train_nr_;
  }
};

}  // namespace motis::loader
