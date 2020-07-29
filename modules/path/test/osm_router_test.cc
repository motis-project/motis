#include "gtest/gtest.h"

#include "motis/path/prepare/osm/osm_graph_contractor.h"
#include "motis/path/prepare/osm/osm_router.h"

#include "./osm_graph_utils.h"

namespace mp = motis::path;

TEST(osm_router, simple) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1, true);
  add_edge(graph, 0, 1, 10);
  set_single_component(graph);

  {
    auto const result = mp::shortest_path_distances(graph, 0, {1});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(10., result.front());
  }
  {
    auto const contracted = mp::contract_graph(graph);
    auto const result = mp::shortest_path_distances(contracted, 0, {1});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(10., result.front());
  }
}

TEST(osm_router, oneway) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1);
  add_node(graph, 2, true);
  add_edge(graph, 0, 1, 10);
  add_edge(graph, 1, 2, 10);
  add_edge(graph, 2, 0, 10);
  set_single_component(graph);

  {
    auto const result = mp::shortest_path_distances(graph, 0, {2});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(20., result.front());
  }
  {
    auto const contracted = mp::contract_graph(graph);
    auto const result = mp::shortest_path_distances(contracted, 0, {2});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(20., result.front());
  }
}

TEST(osm_router, alternative) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1);
  add_node(graph, 2, true);
  add_node(graph, 3, true);
  add_edge(graph, 0, 1, 10);
  add_edge(graph, 1, 2, 10);
  add_edge(graph, 0, 3, 5);
  add_edge(graph, 3, 2, 5);
  set_single_component(graph);

  {
    auto const result = mp::shortest_path_distances(graph, 0, {2});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(10., result.front());
  }
  {
    auto const contracted = mp::contract_graph(graph);
    auto const result = mp::shortest_path_distances(contracted, 0, {2});
    ASSERT_EQ(1, result.size());
    EXPECT_EQ(10., result.front());
  }
}
