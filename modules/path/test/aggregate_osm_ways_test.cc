#include "gtest/gtest.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/path/prepare/osm/osm_way.h"

namespace mp = motis::path;

mp::osm_way make_way(mcd::vector<int64_t> const& ids,
                     mp::source_bits const sb) {
  utl::verify(ids.size() >= 2, "make_ways: invalid input");

  mp::osm_path path{ids.size()};
  path.osm_node_ids_ = ids;
  path.polyline_.resize(ids.size());

  return mp::osm_way{{0}, sb, std::move(path)};
}

mcd::vector<mp::osm_way> make_ways(mcd::vector<mcd::vector<int64_t>> const& in,
                                   bool oneway = false) {
  return mcd ::to_vec(in, [&](auto const& ids) {
    return make_way(
        ids, oneway ? mp::source_bits::ONEWAY : mp::source_bits::NO_SOURCE);
  });
}

#define COUNT_WAY(vec, from_expected, to_expected)                     \
  (std::count_if(begin(vec), end(vec), [](auto const& e) {             \
    return (e.from() == (from_expected) && e.to() == (to_expected)) || \
           (e.from() == (to_expected) && e.to() == (from_expected));   \
  }))

TEST(aggregate_osm_ways, pairwise) {
  {
    auto input = make_ways({{0, 1}, {2, 3}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(make_ways({{0, 1}, {2, 3}}), output);
    EXPECT_EQ(mp::source_bits::NO_SOURCE, output[0].source_bits_);
  }

  {
    auto input = make_ways({{0, 1}, {1, 2}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
  }

  {
    auto input = make_ways({{0, 1}, {2, 1}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
  }

  {
    auto input = make_ways({{1, 0}, {1, 2}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
  }

  {
    auto input = make_ways({{1, 0}, {2, 1}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
  }
}

TEST(aggregate_osm_ways, pairwise_oneway) {
  {
    auto input = make_ways({{0, 1}, {1, 2}}, true);
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
    EXPECT_EQ(mp::source_bits::ONEWAY, output[0].source_bits_);
  }

  {
    auto input = make_ways({{0, 1}, {2, 1}}, true);
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(2, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 1));
    EXPECT_EQ(1, COUNT_WAY(output, 2, 1));
    EXPECT_EQ(mp::source_bits::ONEWAY, output[0].source_bits_);
    EXPECT_EQ(mp::source_bits::ONEWAY, output[1].source_bits_);
  }

  {
    auto input = make_ways({{1, 0}, {1, 2}}, true);
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(2, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 1, 0));
    EXPECT_EQ(1, COUNT_WAY(output, 1, 2));
    EXPECT_EQ(mp::source_bits::ONEWAY, output[0].source_bits_);
    EXPECT_EQ(mp::source_bits::ONEWAY, output[1].source_bits_);
  }

  {
    auto input = make_ways({{1, 0}, {2, 1}}, true);
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(1, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
    EXPECT_EQ(mp::source_bits::ONEWAY, output[0].source_bits_);
  }
}

TEST(aggregate_osm_ways, pairwise_source_bits) {
  {
    auto input =
        mcd::vector<mp::osm_way>{make_way({0, 1}, mp::source_bits::RAIL),
                                 make_way({2, 1}, mp::source_bits::TRAM)};
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(2, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 1, 0));
    EXPECT_EQ(1, COUNT_WAY(output, 1, 2));

    EXPECT_EQ(input[output[0].from() == 0 ? 0 : 1].source_bits_,
              output[0].source_bits_);
    EXPECT_EQ(input[output[1].from() == 2 ? 1 : 0].source_bits_,
              output[1].source_bits_);
  }
  {
    auto input = mcd::vector<mp::osm_way>{
        make_way({0, 1}, mp::source_bits::RAIL | mp::source_bits::ONEWAY),
        make_way({2, 1}, mp::source_bits::RAIL | mp::source_bits::ONEWAY)};
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(2, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 1, 0));
    EXPECT_EQ(1, COUNT_WAY(output, 1, 2));

    EXPECT_EQ(mp::source_bits::RAIL | mp::source_bits::ONEWAY,
              output[0].source_bits_);
    EXPECT_EQ(mp::source_bits::RAIL | mp::source_bits::ONEWAY,
              output[0].source_bits_);
  }
}

TEST(aggregate_osm_ways, three_star) {
  {
    auto input = make_ways({{0, 3}, {1, 3}, {2, 3}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(3, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 3));
    EXPECT_EQ(1, COUNT_WAY(output, 1, 3));
    EXPECT_EQ(1, COUNT_WAY(output, 2, 3));
  }

  {
    auto input = make_ways({{0, 3}, {1, 3}, {2, 3}, {2, 4}});
    auto output = mp::aggregate_osm_ways(input);
    ASSERT_EQ(3, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 3));
    EXPECT_EQ(1, COUNT_WAY(output, 1, 3));
    EXPECT_EQ(1, COUNT_WAY(output, 4, 3));
  }
}

TEST(aggregate_osm_ways, loop) {
  {
    auto input = make_ways({{1, 0},  //
                            {2, 1},  //
                            {3, 2},  // loop part 1
                            {3, 2},  // loop part 2
                            {4, 2},  //
                            {5, 4},  //
                            {5, 6}});
    auto output = mp::aggregate_osm_ways(input);

    ASSERT_EQ(3, output.size());
    EXPECT_EQ(1, COUNT_WAY(output, 0, 2));
    EXPECT_EQ(1, COUNT_WAY(output, 2, 2));  // the loop
    EXPECT_EQ(1, COUNT_WAY(output, 2, 6));
  }
}
