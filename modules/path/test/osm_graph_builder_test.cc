#include "gtest/gtest.h"

#include "motis/path/prepare/osm/osm_graph.h"
#include "motis/path/prepare/osm/osm_graph_builder.h"
#include "motis/path/prepare/schedule/stations.h"

namespace mp = motis::path;

TEST(osm_graph_builder, at_way_node) {
  auto stations =
      mp::make_station_index({{"1", "test", {49.8678310, 8.6781722}}});

  mp::osm_graph graph;
  mp::osm_graph_builder builder(graph, stations);

  mp::osm_path path{{
                        {49.8675940, 8.6785566},  // 0
                        {49.8676567, 8.6784000},  // 1
                        {49.8677067, 8.6783078},  // 2
                        {49.8678310, 8.6781722},  // 3 stop_position
                        {49.8682052, 8.6779246},  // 4
                        {49.8682487, 8.6778950},  // 5
                    },
                    {0L, 1L, 2L, 3L, 4L, 5L}};
  builder.add_component({mp::osm_way{{0}, true, path}});

  ASSERT_EQ(3, graph.nodes_.size());

  auto const& n0 = graph.nodes_.at(0);
  EXPECT_EQ(path.polyline_.at(0), n0->pos_);
  ASSERT_EQ(1, n0->edges_.size());
  auto const& p0 = graph.paths_.at(n0->edges_.at(0).polyline_idx_);
  EXPECT_EQ(path.partial(0, 4), p0);

  auto const& n1 = graph.nodes_.at(1);
  EXPECT_EQ(path.polyline_.at(3), n1->pos_);
  ASSERT_EQ(1, n1->edges_.size());
  auto const& p1 = graph.paths_.at(n1->edges_.at(0).polyline_idx_);
  EXPECT_EQ(path.partial(3, 6), p1);

  auto const& n2 = graph.nodes_.at(2);
  EXPECT_EQ(path.polyline_.at(5), n2->pos_);
  ASSERT_TRUE(n2->edges_.empty());
}

TEST(osm_graph_builder, between_way_nodes) {
  auto stations =
      mp::make_station_index({{"1", "test", {49.86780008585, 8.678289949893}}});

  mp::osm_graph graph;
  mp::osm_graph_builder builder(graph, stations);

  mp::osm_path path{{
                        {49.8675940, 8.6785566},  // 0
                        {49.8676567, 8.6784000},  // 1
                        {49.8677067, 8.6783078},  // 2 between here
                        {49.8678310, 8.6781722},  // 3 and here
                        {49.8682052, 8.6779246},  // 4
                        {49.8682487, 8.6778950},  // 5
                    },
                    {0L, 1L, 2L, 3L, 4L, 5L}};
  builder.add_component({mp::osm_way{{0}, true, path}});

  ASSERT_EQ(3, graph.nodes_.size());

  auto const& n0 = graph.nodes_.at(0);
  EXPECT_EQ(path.polyline_.at(0), n0->pos_);
  ASSERT_EQ(1, n0->edges_.size());
  auto const& p0 = graph.paths_.at(n0->edges_.at(0).polyline_idx_);
  ASSERT_EQ(4, p0.size());
  EXPECT_EQ(path.partial(0, 3), p0.partial(0, 3));
  EXPECT_EQ(-1, p0.osm_node_ids_.at(3));

  auto const& n1 = graph.nodes_.at(1);
  EXPECT_EQ(-1, n1->osm_id_);
  ASSERT_EQ(1, n1->edges_.size());
  auto const& p1 = graph.paths_.at(n1->edges_.at(0).polyline_idx_);
  ASSERT_EQ(4, p1.size());
  EXPECT_EQ(path.partial(3, 6), p1.partial(1, 4));
  EXPECT_EQ(-1, p1.osm_node_ids_.at(0));

  auto const& n2 = graph.nodes_.at(2);
  EXPECT_EQ(path.polyline_.at(5), n2->pos_);
  ASSERT_TRUE(n2->edges_.empty());
}
