#include "gtest/gtest.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/path/prepare/osm/osm_way.h"

namespace mp = motis::path;

mcd::vector<mp::osm_way> make_ways(mcd::vector<mcd::vector<int64_t>> const& in,
                                   bool oneway = false) {
  return mcd ::to_vec(in, [&](auto const& ids) {
    utl::verify(ids.size() >= 2, "make_ways: invalid input");

    mp::osm_path path{ids.size()};
    path.osm_node_ids_ = ids;
    path.polyline_.resize(ids.size());

    return mp::osm_way{{0}, oneway, std::move(path)};
  });
}

#define COUNT_WAY(vec, from_expected, to_expected)                     \
  (std::count_if(begin(vec), end(vec), [](auto const& e) {             \
    return (e.from() == (from_expected) && e.to() == (to_expected)) || \
           (e.from() == (to_expected) && e.to() == (from_expected));   \
  }))

TEST(aggregate_osm_ways, pairwise) {
  {
    auto ways = make_ways({{0, 1}, {2, 3}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(make_ways({{0, 1}, {2, 3}}), ways);
  }

  {
    auto ways = make_ways({{0, 1}, {1, 2}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }

  {
    auto ways = make_ways({{0, 1}, {2, 1}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }

  {
    auto ways = make_ways({{1, 0}, {1, 2}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }

  {
    auto ways = make_ways({{1, 0}, {2, 1}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }
}

TEST(aggregate_osm_ways, pairwise_oneway) {
  {
    auto ways = make_ways({{0, 1}, {1, 2}}, true);
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }

  {
    auto ways = make_ways({{0, 1}, {2, 1}}, true);
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(2, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 1));
    EXPECT_EQ(1, COUNT_WAY(ways, 2, 1));
  }

  {
    auto ways = make_ways({{1, 0}, {1, 2}}, true);
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(2, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 1, 0));
    EXPECT_EQ(1, COUNT_WAY(ways, 1, 2));
  }

  {
    auto ways = make_ways({{1, 0}, {2, 1}}, true);
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(1, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
  }
}

TEST(aggregate_osm_ways, three_star) {
  {
    auto ways = make_ways({{0, 3}, {1, 3}, {2, 3}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(3, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 3));
    EXPECT_EQ(1, COUNT_WAY(ways, 1, 3));
    EXPECT_EQ(1, COUNT_WAY(ways, 2, 3));
  }

  {
    auto ways = make_ways({{0, 3}, {1, 3}, {2, 3}, {2, 4}});
    mp::aggregate_osm_ways(ways);
    ASSERT_EQ(3, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 3));
    EXPECT_EQ(1, COUNT_WAY(ways, 1, 3));
    EXPECT_EQ(1, COUNT_WAY(ways, 4, 3));
  }
}

TEST(aggregate_osm_ways, loop) {
  {
    auto ways = make_ways({{1, 0},  //
                           {2, 1},  //
                           {3, 2},  // loop part 1
                           {3, 2},  // loop part 2
                           {4, 2},  //
                           {5, 4},  //
                           {5, 6}});
    mp::aggregate_osm_ways(ways);

    ASSERT_EQ(3, ways.size());
    EXPECT_EQ(1, COUNT_WAY(ways, 0, 2));
    EXPECT_EQ(1, COUNT_WAY(ways, 2, 2));  // the loop
    EXPECT_EQ(1, COUNT_WAY(ways, 2, 6));
  }
}
