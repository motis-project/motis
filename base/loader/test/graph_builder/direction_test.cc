#include "gtest/gtest.h"

#include "motis/core/common/date_time_util.h"

#include "./graph_builder_test.h"

namespace motis::loader {

class loader_direction_services_graph_builder_test
    : public loader_graph_builder_test {
public:
  loader_direction_services_graph_builder_test()
      : loader_graph_builder_test("direction-services", "20150911", 2) {}
};

TEST_F(loader_direction_services_graph_builder_test, direction_station) {
  // Get route starting at Euskirchen
  auto node_it = std::find_if(
      begin(sched_->route_index_to_first_route_node_),
      end(sched_->route_index_to_first_route_node_), [&](node const* n) {
        return sched_->stations_[n->get_station()->id_]->eva_nr_ == "8000100";
      });
  ASSERT_FALSE(node_it == end(sched_->route_index_to_first_route_node_));

  auto connections = get_connections(*node_it, time{0});
  ASSERT_GE(connections.size(), 16);

  for (auto i = 0U; i < 12; ++i) {
    auto con_info = std::get<0>(connections[i])->full_con_->con_info_;
    ASSERT_FALSE(con_info->dir_ == nullptr);
    ASSERT_EQ("Kreuzberg(Ahr)", *con_info->dir_);  // NOLINT
  }

  for (auto i = 12UL; i < connections.size(); ++i) {
    auto con_info = std::get<0>(connections[i])->full_con_->con_info_;
    ASSERT_TRUE(con_info->dir_ == nullptr);
  }
}

TEST_F(loader_direction_services_graph_builder_test, direction_text) {
  // Get route starting at Wissmar Gewerbegebiet
  auto node_it = std::find_if(
      begin(sched_->route_index_to_first_route_node_),
      end(sched_->route_index_to_first_route_node_), [&](node const* n) {
        return sched_->stations_[n->get_station()->id_]->eva_nr_ == "0114965";
      });
  ASSERT_FALSE(node_it == end(sched_->route_index_to_first_route_node_));

  auto connections = get_connections(*node_it, time{0});
  ASSERT_GE(connections.size(), 27);

  for (auto const& e : connections) {
    auto con_info = std::get<0>(e)->full_con_->con_info_;
    ASSERT_FALSE(con_info->dir_ == nullptr);
    ASSERT_EQ("Krofdorf-Gleiberg Evangelische Ki",
              *con_info->dir_);  // NOLINT
  }
}

}  // namespace motis::loader
