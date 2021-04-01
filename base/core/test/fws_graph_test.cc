#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <iterator>

#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"

#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/core/common/fws_graph.h"

using ::testing::ElementsAreArray;

namespace motis {

namespace {

template <typename Node, typename Edge>
void print_graph(fws_graph<Node, Edge>& g) {
  for (auto [i, node] : utl::enumerate(g.nodes_)) {
    std::cout << "node: " << node << "\n";
    std::cout << "  outgoing edges:\n";
    for (auto const& e : g.outgoing_edges(i)) {
      std::cout << "    " << e << "\n";
    }
    std::cout << "  incoming edges:\n";
    for (auto const& e : g.incoming_edges(i)) {
      std::cout << "    " << e << "\n";
    }
  }
}

struct test_node {
  CISTA_COMPARABLE()
  std::uint32_t id_{};
  std::uint32_t tag_{};
};

struct test_edge {
  CISTA_COMPARABLE()
  std::uint32_t from_{};
  std::uint32_t to_{};
  std::uint32_t weight_{};
};

inline std::ostream& operator<<(std::ostream& out, test_node const& n) {
  return out << "{id=" << n.id_ << ", tag=" << n.tag_ << "}";
}

inline std::ostream& operator<<(std::ostream& out, test_edge const& e) {
  return out << "{from=" << e.from_ << ", to=" << e.to_
             << ", weight=" << e.weight_ << "}";
}

template <typename Node, typename Edge>
void check_graph(
    fws_graph<Node, Edge>& g,
    mcd::hash_map<std::uint32_t, mcd::vector<Edge>> const& check_fwd,
    mcd::hash_map<std::uint32_t, mcd::vector<Edge>> const& check_bwd) {
  for (auto const& [from, edges] : check_fwd) {
    auto const bucket = g.outgoing_edges(from);
    EXPECT_EQ(edges.size(), bucket.size());
    EXPECT_THAT(bucket, ElementsAreArray(edges));
  }
  for (auto const& [to, edges] : check_bwd) {
    auto const bucket = g.incoming_edges(to);
    EXPECT_EQ(edges.size(), bucket.size());
    EXPECT_THAT(bucket, ElementsAreArray(edges));
  }
}

template <typename Node, typename Edge>
void add_edge(fws_graph<Node, Edge>& g,
              mcd::hash_map<std::uint32_t, mcd::vector<Edge>>& check_fwd,
              mcd::hash_map<std::uint32_t, mcd::vector<Edge>>& check_bwd,
              Edge const& e) {
  g.push_back_edge(e);
  check_fwd[e.from_].push_back(e);
  check_bwd[e.to_].push_back(e);

  check_graph(g, check_fwd, check_bwd);
}

}  // namespace

TEST(fws_graph_test, t1) {
  fws_graph<test_node, test_edge> g;
  mcd::hash_map<std::uint32_t, mcd::vector<test_edge>> check_fwd;
  mcd::hash_map<std::uint32_t, mcd::vector<test_edge>> check_bwd;

  g.nodes_.emplace_back(0U, 4U);
  g.nodes_.emplace_back(1U, 8U);
  g.nodes_.emplace_back(2U, 15U);
  g.nodes_.emplace_back(3U, 16U);
  g.nodes_.emplace_back(4U, 23U);
  g.nodes_.emplace_back(5U, 42U);
  g.nodes_.emplace_back(6U, 42U);
  g.nodes_.emplace_back(7U, 42U);
  g.nodes_.emplace_back(8U, 42U);

  add_edge(g, check_fwd, check_bwd, {0U, 2U, 5U});
  add_edge(g, check_fwd, check_bwd, {0U, 3U, 7U});
  add_edge(g, check_fwd, check_bwd, {2U, 1U, 3U});
  add_edge(g, check_fwd, check_bwd, {3U, 2U, 2U});
  add_edge(g, check_fwd, check_bwd, {2U, 0U, 5U});
  add_edge(g, check_fwd, check_bwd, {0U, 4U, 4U});
  add_edge(g, check_fwd, check_bwd, {2U, 5U, 20U});
  add_edge(g, check_fwd, check_bwd, {0U, 6U, 1U});
  add_edge(g, check_fwd, check_bwd, {0U, 7U, 1U});
  add_edge(g, check_fwd, check_bwd, {0U, 8U, 1U});
  add_edge(g, check_fwd, check_bwd, {2U, 3U, 1U});
  add_edge(g, check_fwd, check_bwd, {2U, 4U, 1U});

  auto g_copy = g;
  check_graph(g, check_fwd, check_bwd);

  if (HasFailure()) {
    print_graph(g);
  }
}

}  // namespace motis
