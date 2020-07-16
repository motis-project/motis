#include "gtest/gtest.h"

#include "motis/path/prepare/osm/osm_graph_contractor.h"

#include "./osm_graph_utils.h"

namespace mp = motis::path;

std::vector<mp::osm_graph_dist> get_distances(mp::osm_graph const& graph) {
  mp::osm_graph_contractor contractor{graph};
  contractor.contract();
  return contractor.collect_distances();
}

TEST(osm_graph_contractor, unconnected) {
  mp::osm_graph graph;
  ASSERT_TRUE(get_distances(graph).empty());

  {
    add_node(graph, 0);
    set_single_component(graph);

    ASSERT_TRUE(get_distances(graph).empty());
  }
  {
    add_node(graph, 1, true);
    set_single_component(graph);

    ASSERT_TRUE(get_distances(graph).empty());
  }
  {
    add_node(graph, 2, true);
    set_single_component(graph);

    ASSERT_TRUE(get_distances(graph).empty());
  }
}

TEST(osm_graph_contractor, direct) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1, true);
  add_edge(graph, 0, 1, 10);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(1, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 1, 10}), result.at(0));
  }

  add_edge(graph, 1, 0, 5);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(2, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 1, 10}), result.at(0));
    EXPECT_EQ((mp::osm_graph_dist{1, 0, 5}), result.at(1));
  }
}

TEST(osm_graph_contractor, single) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1);
  add_node(graph, 2, true);
  add_edge(graph, 0, 1, 5);
  add_edge(graph, 1, 2, 3);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(1, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 2, 8}), result.at(0));
  }

  add_edge(graph, 0, 2, 10);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(1, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 2, 8}), result.at(0));
  }

  add_edge(graph, 0, 2, 5);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(1, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 2, 5}), result.at(0));
  }
}

TEST(osm_graph_contractor, two_terminals_complex) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1);
  add_node(graph, 2);
  add_node(graph, 3);
  add_node(graph, 4);
  add_node(graph, 5);
  add_node(graph, 6, true);

  add_edge(graph, 0, 1, 1);  // *
  add_edge(graph, 1, 2, 1);  // *
  add_edge(graph, 1, 3, 4);
  add_edge(graph, 2, 3, 2);
  add_edge(graph, 2, 4, 2);  // *
  add_edge(graph, 3, 4, 3);
  add_edge(graph, 4, 5, 1);  // *
  add_edge(graph, 4, 6, 3);
  add_edge(graph, 5, 6, 1);  // *
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(1, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 6, 6}), result.at(0));
  }
}

TEST(osm_graph_contractor, tree_terminals_asymetric_star) {
  mp::osm_graph graph;

  add_node(graph, 0, true);
  add_node(graph, 1, true);
  add_node(graph, 2, true);
  add_node(graph, 3);  // center

  add_edge(graph, 0, 3, 4);
  add_edge(graph, 3, 0, 5);
  add_edge(graph, 1, 3, 7);
  add_edge(graph, 3, 1, 6);
  add_edge(graph, 2, 3, 10);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(4, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 1, 4 + 6}), result.at(0));
    EXPECT_EQ((mp::osm_graph_dist{1, 0, 7 + 5}), result.at(1));
    EXPECT_EQ((mp::osm_graph_dist{2, 0, 10 + 5}), result.at(2));
    EXPECT_EQ((mp::osm_graph_dist{2, 1, 10 + 6}), result.at(3));
  }
}

TEST(osm_graph_contractor, four_terminals_single_railroad) {
  mp::osm_graph graph;

  // 0-4\      /6-2
  //     8-9-10
  // 1-5/      \7-3

  add_node(graph, 0, true);
  add_node(graph, 1, true);
  add_node(graph, 2, true);
  add_node(graph, 3, true);

  add_node(graph, 4);
  add_node(graph, 5);
  add_node(graph, 6);
  add_node(graph, 7);

  add_node(graph, 8);
  add_node(graph, 9);
  add_node(graph, 10);

  add_edge2(graph, 0, 4, 1);
  add_edge2(graph, 1, 5, 1);
  add_edge2(graph, 2, 6, 1);
  add_edge2(graph, 3, 7, 1);

  add_edge2(graph, 4, 8, 1);
  add_edge2(graph, 5, 8, 1);
  add_edge2(graph, 6, 10, 1);
  add_edge2(graph, 7, 10, 1);

  add_edge2(graph, 8, 9, 1);
  add_edge2(graph, 9, 10, 1);
  set_single_component(graph);

  {
    auto const result = get_distances(graph);
    ASSERT_EQ(12, result.size());
    EXPECT_EQ((mp::osm_graph_dist{0, 1, 4}), result.at(0));
    EXPECT_EQ((mp::osm_graph_dist{0, 2, 6}), result.at(1));
    EXPECT_EQ((mp::osm_graph_dist{0, 3, 6}), result.at(2));
    EXPECT_EQ((mp::osm_graph_dist{1, 0, 4}), result.at(3));
    EXPECT_EQ((mp::osm_graph_dist{1, 2, 6}), result.at(4));
    EXPECT_EQ((mp::osm_graph_dist{1, 3, 6}), result.at(5));
    EXPECT_EQ((mp::osm_graph_dist{2, 0, 6}), result.at(6));
    EXPECT_EQ((mp::osm_graph_dist{2, 1, 6}), result.at(7));
    EXPECT_EQ((mp::osm_graph_dist{2, 3, 4}), result.at(8));
    EXPECT_EQ((mp::osm_graph_dist{3, 0, 6}), result.at(9));
    EXPECT_EQ((mp::osm_graph_dist{3, 1, 6}), result.at(10));
    EXPECT_EQ((mp::osm_graph_dist{3, 2, 4}), result.at(11));
  }
}
