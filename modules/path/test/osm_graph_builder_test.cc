#include "gtest/gtest.h"

#include "utl/equal_ranges_linear.h"
#include "utl/parser/csv.h"
#include "utl/parser/file.h"
#include "utl/to_set.h"

#include "motis/path/prepare/osm/osm_graph.h"
#include "motis/path/prepare/osm/osm_graph_builder.h"
#include "motis/path/prepare/schedule/stations.h"

namespace mp = motis::path;

TEST(osm_graph_builder, at_way_node) {
  auto stations =
      mp::make_station_index({{"1", "test", {49.8678310, 8.6781722}}});

  mp::osm_graph graph;
  mp::osm_graph_builder builder(
      graph,
      mp::source_spec{mp::source_spec::category::UNKNOWN,
                      mp::source_spec::router::OSM_REL},
      stations);

  mp::osm_path path{{
                        {49.8675940, 8.6785566},  // 0
                        {49.8676567, 8.6784000},  // 1
                        {49.8677067, 8.6783078},  // 2
                        {49.8678310, 8.6781722},  // 3 stop_position
                        {49.8682052, 8.6779246},  // 4
                        {49.8682487, 8.6778950},  // 5
                    },
                    {0L, 1L, 2L, 3L, 4L, 5L}};
  builder.add_component({mp::osm_way{{0}, mp::source_bits::ONEWAY, path}});

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
  mp::osm_graph_builder builder(
      graph,
      mp::source_spec{mp::source_spec::category::UNKNOWN,
                      mp::source_spec::router::OSM_REL},
      stations);

  mp::osm_path path{{
                        {49.8675940, 8.6785566},  // 0
                        {49.8676567, 8.6784000},  // 1
                        {49.8677067, 8.6783078},  // 2 between here
                        {49.8678310, 8.6781722},  // 3 and here
                        {49.8682052, 8.6779246},  // 4
                        {49.8682487, 8.6778950},  // 5
                    },
                    {0L, 1L, 2L, 3L, 4L, 5L}};
  builder.add_component({mp::osm_way{{0}, mp::source_bits::ONEWAY, path}});

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

enum { way_idx, node_id, lat, lng };
using csv_node = std::tuple<int64_t, int64_t, double, double>;
static const utl::column_mapping<csv_node> columns = {
    {"way_idx", "node_id", "lat", "lng"}};

TEST(osm_graph_builder, heusenstamm) {
  auto stations = mp::make_station_index(
      {{"0", "bieber", {50.090555000000, 8.808381000000}},
       {"1", "heusenstamm", {50.060461000000, 8.801364000000}},
       {"2", "d-steinberg", {50.023309000000, 8.793898000000}},
       {"3", "d-mitte", {50.017860000000, 8.788452000000}},
       {"4", "d-bahnhofheusenstamm", {50.060461000000, 8.801364000000}},
       {"5", "d-bahnhof", {50.007974000000, 8.785091000000}}});

  mp::osm_graph graph;
  mp::osm_graph_builder builder(
      graph,
      mp::source_spec{mp::source_spec::category::UNKNOWN,
                      mp::source_spec::router::OSM_REL},
      stations);

  {
    auto const csv_nodes = utl::read_file<csv_node>(
        std::string{"modules/path/test_resources/heusenstamm.csv"}, columns);

    mcd::vector<mp::osm_way> ways;
    utl::equal_ranges_linear(
        csv_nodes,
        [](auto const& lhs, auto const& rhs) {
          return std::get<way_idx>(lhs) == std::get<way_idx>(rhs);
        },
        [&](auto lb, auto ub) {
          ways.emplace_back();
          ways.back().ids_.push_back(std::get<way_idx>(*lb));
          ways.back().path_ = mp::osm_path{
              mcd::to_vec(
                  lb, ub,
                  [](auto const& t) {
                    return geo::latlng{std::get<lat>(t), std::get<lng>(t)};
                  }),
              mcd::to_vec(lb, ub,
                          [](auto const& t) { return std::get<node_id>(t); })};
        });

    builder.add_component(ways);
  }

  {
    auto const station_links =
        utl::to_set(graph.node_station_links_,
                    [](auto const& nsl) { return nsl.station_id_; });
    EXPECT_EQ(6, station_links.size());
  }
}

TEST(osm_graph_builder, dornach) {
  auto stations = mp::make_station_index({{"0", "0", {47.48581, 7.606187}}});

  mp::osm_path path{{
                        {47.4881248, 7.6091958},
                        {47.4880832, 7.6092541},
                        {47.4880828, 7.6092546},
                        {47.4880214, 7.6093681},
                    },
                    {1796922120, 7502748777, 304541648, 304541645}};

  mp::osm_graph graph;
  mp::osm_graph_builder builder(
      graph,
      mp::source_spec{mp::source_spec::category::UNKNOWN,
                      mp::source_spec::router::OSM_REL},
      stations);
  builder.add_component({mp::osm_way{{0}, mp::source_bits::NO_SOURCE, path}});
  EXPECT_EQ(1, graph.components_);
}
