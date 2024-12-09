#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include "motis/gbfs/partition.h"

using namespace motis::gbfs;

// helper function to compare sets regardless of order
template <typename T>
bool compare_partitions(std::vector<std::vector<T>> const& actual,
                        std::vector<std::vector<T>> const& expected) {
  if (actual.size() != expected.size()) {
    return false;
  }

  auto actual_sets = std::vector<std::set<T>>{};
  auto expected_sets = std::vector<std::set<T>>{};

  for (auto const& vec : actual) {
    actual_sets.emplace_back(begin(vec), end(vec));
  }
  for (auto const& vec : expected) {
    expected_sets.emplace_back(begin(vec), end(vec));
  }

  std::sort(actual_sets.begin(), actual_sets.end());
  std::sort(expected_sets.begin(), expected_sets.end());

  return actual_sets == expected_sets;
}

TEST(motis, gbfs_partition_test) {
  using T = int;

  auto p = partition<T>{10};

  // Initial partition test
  auto const single_set =
      std::vector<std::vector<T>>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
  EXPECT_TRUE(compare_partitions(p.get_sets(), single_set));

  // First refinement
  p.refine(std::array{1, 2, 3});
  auto const two_sets =
      std::vector<std::vector<T>>{{1, 2, 3}, {0, 4, 5, 6, 7, 8, 9}};
  EXPECT_TRUE(compare_partitions(p.get_sets(), two_sets));

  // Same refinement in different orders should yield same result
  p.refine(std::array{1, 2, 3});
  EXPECT_TRUE(compare_partitions(p.get_sets(), two_sets));
  p.refine(std::array{3, 2, 1});
  EXPECT_TRUE(compare_partitions(p.get_sets(), two_sets));

  // Further refinement
  p.refine(std::array{7, 8});
  auto const three_sets =
      std::vector<std::vector<T>>{{1, 2, 3}, {7, 8}, {0, 4, 5, 6, 9}};
  EXPECT_TRUE(compare_partitions(p.get_sets(), three_sets));

  // Final refinement
  p.refine(std::array{1, 3});
  auto const four_sets =
      std::vector<std::vector<T>>{{1, 3}, {2}, {7, 8}, {0, 4, 5, 6, 9}};
  EXPECT_TRUE(compare_partitions(p.get_sets(), four_sets));
}

TEST(motis, gbfs_partition_empty_refinement) {
  using T = int;
  auto p = partition<T>{5};

  p.refine(std::array<T, 0>{});
  EXPECT_TRUE(compare_partitions(p.get_sets(),
                                 std::vector<std::vector<T>>{{0, 1, 2, 3, 4}}));
}

TEST(motis, gbfs_partition_single_element) {
  using T = int;
  auto p = partition<T>{1};

  p.refine(std::array{0});
  EXPECT_TRUE(
      compare_partitions(p.get_sets(), std::vector<std::vector<T>>{{0}}));
}

TEST(motis, gbfs_partition_disjoint_refinements) {
  using T = int;
  auto p = partition<T>{6};

  p.refine(std::array{0, 1});
  p.refine(std::array{2, 3});
  p.refine(std::array{4, 5});

  auto const expected = std::vector<std::vector<T>>{{0, 1}, {2, 3}, {4, 5}};
  EXPECT_TRUE(compare_partitions(p.get_sets(), expected));
}
