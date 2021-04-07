#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <iterator>

#include "motis/core/common/dynamic_fws_multimap.h"

namespace motis {

namespace {

template <typename T, bool ConstBucket>
void check_result(
    std::vector<T> const& ref,
    typename dynamic_fws_multimap<T>::template bucket<ConstBucket> const&
        result) {
  if (ref.size() != result.size() && result.size() < 10) {
    std::cout << "Invalid result:\n  Expected: ";
    std::copy(begin(ref), end(ref), std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n  Result:   ";
    std::copy(begin(result), end(result),
              std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
  }
  ASSERT_EQ(ref.size(), result.size());
  for (auto i = 0UL; i < ref.size(); ++i) {
    EXPECT_EQ(ref[i], result[i]);
  }
}

template <typename T, typename SizeType>
void print_multimap(dynamic_fws_multimap<T, SizeType>& mm) {
  for (auto const& bucket : mm) {
    std::cout << "key={" << bucket.index() << "} => [ ";
    for (auto const& entry : bucket) {
      std::cout << entry << " ";
    }
    std::cout << "]\n";
  }
}

/*
struct test_node {
  std::uint32_t id_{};
  std::uint32_t tag_{};
};
*/

struct test_edge {
  std::uint32_t from_{};
  std::uint32_t to_{};
  std::uint32_t weight_{};
};

inline std::ostream& operator<<(std::ostream& out, test_edge const& e) {
  return out << "{from=" << e.from_ << ", to=" << e.to_
             << ", weight=" << e.weight_ << "}";
}

}  // namespace

TEST(fws_dynamic_multimap_test, int_1) {
  dynamic_fws_multimap<int> mm;

  ASSERT_EQ(0, mm.element_count());
  ASSERT_EQ(0, mm.index_size());

  mm[0].push_back(42);
  ASSERT_EQ(1, mm.element_count());
  ASSERT_EQ(1, mm.index_size());
  check_result<int>({42}, mm[0]);
  ASSERT_EQ(1, mm[0].size());

  mm[1].push_back(4);
  ASSERT_EQ(2, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  check_result<int>({42}, mm[0]);
  ASSERT_EQ(1, mm[0].size());
  check_result<int>({4}, mm[1]);
  ASSERT_EQ(1, mm[1].size());

  mm[1].push_back(8);
  ASSERT_EQ(3, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  check_result<int>({42}, mm[0]);
  ASSERT_EQ(1, mm[0].size());
  check_result<int>({4, 8}, mm[1]);
  ASSERT_EQ(2, mm[1].size());

  mm[1].emplace_back(15);
  ASSERT_EQ(4, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  check_result<int>({42}, mm[0]);
  ASSERT_EQ(1, mm[0].size());
  check_result<int>({4, 8, 15}, mm[1]);
  ASSERT_EQ(3, mm[1].size());

  mm[1].push_back(16);
  ASSERT_EQ(5, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  check_result<int>({42}, mm[0]);
  ASSERT_EQ(1, mm[0].size());
  check_result<int>({4, 8, 15, 16}, mm[1]);
  ASSERT_EQ(4, mm[1].size());

  mm[0].push_back(23);
  ASSERT_EQ(6, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  check_result<int>({42, 23}, mm[0]);
  ASSERT_EQ(2, mm[0].size());
  check_result<int>({4, 8, 15, 16}, mm[1]);
  ASSERT_EQ(4, mm[1].size());

  print_multimap(mm);
}

TEST(fws_dynamic_multimap_test, graph_1) {
  dynamic_fws_multimap<test_edge> mm;

  mm[0].emplace_back(0U, 1U, 10U);
  mm[1].emplace_back(1U, 2U, 20U);
  mm[1].emplace_back(1U, 3U, 30U);
  mm[3].emplace_back(3U, 0U, 50U);
  mm[2].emplace_back(2U, 3U, 5U);

  print_multimap(mm);
}

}  // namespace motis
