#include <iterator>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cista/reflection/comparable.h"

#include "utl/erase.h"
#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/dynamic_fws_multimap.h"

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

namespace motis {

namespace {

/*
struct test_node {
  std::uint32_t id_{};
  std::uint32_t tag_{};
};
*/

struct test_edge {
  CISTA_COMPARABLE()

  std::uint32_t from_{};
  std::uint32_t to_{};
  std::uint32_t weight_{};
};

inline std::ostream& operator<<(std::ostream& out, test_edge const& e) {
  return out << "{from=" << e.from_ << ", to=" << e.to_
             << ", weight=" << e.weight_ << "}";
}

inline dynamic_fws_multimap<int> build_test_map_1() {
  dynamic_fws_multimap<int> mm;

  mm[0].push_back(4);
  mm[0].push_back(8);

  mm[1].push_back(15);
  mm[1].push_back(16);
  mm[1].push_back(23);
  mm[1].push_back(42);

  mm[2].push_back(100);
  mm[2].push_back(200);
  mm[2].push_back(250);
  mm[2].push_back(300);
  mm[2].push_back(350);
  mm[2].push_back(400);

  return mm;
}

}  // namespace

TEST(dynamic_fws_multimap_test, int_1) {
  dynamic_fws_multimap<int> mm;

  ASSERT_EQ(0, mm.element_count());
  ASSERT_EQ(0, mm.index_size());

  mm[0].push_back(42);
  ASSERT_EQ(1, mm.element_count());
  ASSERT_EQ(1, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42}));
  EXPECT_EQ(1, mm[0].size());

  mm[1].push_back(4);
  ASSERT_EQ(2, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42}));
  EXPECT_EQ(1, mm[0].size());
  EXPECT_THAT(mm[1], ElementsAreArray({4}));
  EXPECT_EQ(1, mm[1].size());

  mm[1].push_back(8);
  ASSERT_EQ(3, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42}));
  EXPECT_EQ(1, mm[0].size());
  EXPECT_THAT(mm[1], ElementsAreArray({4, 8}));
  EXPECT_EQ(2, mm[1].size());

  mm[1].emplace_back(15);
  ASSERT_EQ(4, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42}));
  EXPECT_EQ(1, mm[0].size());
  EXPECT_THAT(mm[1], ElementsAreArray({4, 8, 15}));
  EXPECT_EQ(3, mm[1].size());

  mm[1].push_back(16);
  ASSERT_EQ(5, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42}));
  EXPECT_EQ(1, mm[0].size());
  EXPECT_THAT(mm[1], ElementsAreArray({4, 8, 15, 16}));
  EXPECT_EQ(4, mm[1].size());

  mm[0].push_back(23);
  ASSERT_EQ(6, mm.element_count());
  ASSERT_EQ(2, mm.index_size());
  EXPECT_THAT(mm[0], ElementsAreArray({42, 23}));
  EXPECT_EQ(2, mm[0].size());
  EXPECT_THAT(mm[1], ElementsAreArray({4, 8, 15, 16}));
  EXPECT_EQ(4, mm[1].size());

  // begin / end
  EXPECT_THAT(std::vector<int>(mm[0].begin(), mm[0].end()),
              ElementsAreArray({42, 23}));
  EXPECT_THAT(std::vector<int>(mm[1].begin(), mm[1].end()),
              ElementsAreArray({4, 8, 15, 16}));
  EXPECT_THAT(std::vector<int>(std::begin(mm[0]), std::end(mm[0])),
              ElementsAreArray({42, 23}));
  EXPECT_THAT(std::vector<int>(std::begin(mm[1]), std::end(mm[1])),
              ElementsAreArray({4, 8, 15, 16}));
  EXPECT_THAT(std::vector<int>(begin(mm[0]), end(mm[0])),
              ElementsAreArray({42, 23}));
  EXPECT_THAT(std::vector<int>(begin(mm[1]), end(mm[1])),
              ElementsAreArray({4, 8, 15, 16}));

  // cbegin / cend
  EXPECT_THAT(std::vector<int>(mm[0].cbegin(), mm[0].cend()),
              ElementsAreArray({42, 23}));
  EXPECT_THAT(std::vector<int>(mm[1].cbegin(), mm[1].cend()),
              ElementsAreArray({4, 8, 15, 16}));
  EXPECT_THAT(std::vector<int>(std::cbegin(mm[0]), std::cend(mm[0])),
              ElementsAreArray({42, 23}));
  EXPECT_THAT(std::vector<int>(std::cbegin(mm[1]), std::cend(mm[1])),
              ElementsAreArray({4, 8, 15, 16}));

  // rbegin / rend
  EXPECT_THAT(std::vector<int>(mm[0].rbegin(), mm[0].rend()),
              ElementsAreArray({23, 42}));
  EXPECT_THAT(std::vector<int>(mm[1].rbegin(), mm[1].rend()),
              ElementsAreArray({16, 15, 8, 4}));
  EXPECT_THAT(std::vector<int>(std::rbegin(mm[0]), std::rend(mm[0])),
              ElementsAreArray({23, 42}));
  EXPECT_THAT(std::vector<int>(std::rbegin(mm[1]), std::rend(mm[1])),
              ElementsAreArray({16, 15, 8, 4}));
  EXPECT_THAT(std::vector<int>(rbegin(mm[0]), rend(mm[0])),
              ElementsAreArray({23, 42}));
  EXPECT_THAT(std::vector<int>(rbegin(mm[1]), rend(mm[1])),
              ElementsAreArray({16, 15, 8, 4}));

  // crbegin / crend
  EXPECT_THAT(std::vector<int>(mm[0].crbegin(), mm[0].crend()),
              ElementsAreArray({23, 42}));
  EXPECT_THAT(std::vector<int>(mm[1].crbegin(), mm[1].crend()),
              ElementsAreArray({16, 15, 8, 4}));
  EXPECT_THAT(std::vector<int>(std::crbegin(mm[0]), std::crend(mm[0])),
              ElementsAreArray({23, 42}));
  EXPECT_THAT(std::vector<int>(std::crbegin(mm[1]), std::crend(mm[1])),
              ElementsAreArray({16, 15, 8, 4}));
}

TEST(dynamic_fws_multimap_test, graph_1) {
  dynamic_fws_multimap<test_edge> mm;

  mm[0].emplace_back(0U, 1U, 10U);
  mm[1].emplace_back(1U, 2U, 20U);
  mm[1].emplace_back(1U, 3U, 30U);
  mm[3].emplace_back(3U, 0U, 50U);
  mm[2].emplace_back(2U, 3U, 5U);

  ASSERT_EQ(4, mm.index_size());
  EXPECT_EQ(5, mm.element_count());

  EXPECT_THAT(mm[0], ElementsAreArray({test_edge{0U, 1U, 10U}}));
  EXPECT_THAT(mm[1], ElementsAreArray(
                         {test_edge{1U, 2U, 20U}, test_edge{1U, 3U, 30U}}));
  EXPECT_THAT(mm[2], ElementsAreArray({test_edge{2U, 3U, 5U}}));
  EXPECT_THAT(mm[3], ElementsAreArray({test_edge{3U, 0U, 50U}}));
}

TEST(dynamic_fws_multimap_test, int_2) {
  auto const mm = build_test_map_1();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(12, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));

  EXPECT_THAT(mm.front(), ElementsAreArray({4, 8}));
  EXPECT_THAT(mm.back(), ElementsAreArray({100, 200, 250, 300, 350, 400}));
  EXPECT_EQ(15, mm[1].front());
  EXPECT_EQ(42, mm[1].back());
}

TEST(dynamic_fws_multimap_test, int_insert_1) {
  auto mm = build_test_map_1();

  mm[1].insert(std::next(mm[1].begin(), 2), 20);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(13, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 20, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));
}

TEST(dynamic_fws_multimap_test, int_insert_2) {
  auto mm = build_test_map_1();

  auto const val = 20;
  mm[1].insert(std::next(mm[1].begin(), 2), val);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(13, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 20, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));
}

TEST(dynamic_fws_multimap_test, int_erase_1) {
  auto mm = build_test_map_1();

  utl::erase(mm[1], 16);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(11, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));

  utl::erase(mm[2], 100);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(10, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 250, 300, 350, 400}));

  utl::erase(mm[2], 400);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(9, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 250, 300, 350}));

  utl::erase(mm[2], 250);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 300, 350}));

  utl::erase(mm[1], 404);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 300, 350}));
}

TEST(dynamic_fws_multimap_test, int_erase_2) {
  auto mm = build_test_map_1();

  utl::erase_if(mm[2], [](int e) { return e % 100 == 0; });

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({250, 350}));
}

TEST(dynamic_fws_multimap_test, int_erase_3) {
  auto mm = build_test_map_1();

  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  auto const it1 = mm[1].erase(std::next(mm[1].begin(), 1));
  ASSERT_EQ(it1, std::next(mm[1].begin(), 1));

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(11, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));

  auto const it2 = mm[2].erase(mm[2].begin());
  ASSERT_EQ(it2, mm[2].begin());

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(10, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 250, 300, 350, 400}));

  auto const it3 = mm[2].erase(std::next(mm[2].begin(), 4));
  ASSERT_EQ(it3, mm[2].end());

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(9, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 250, 300, 350}));

  auto const it4 = mm[2].erase(std::next(mm[2].begin(), 1));
  ASSERT_EQ(it4, std::next(mm[2].begin(), 1));

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({200, 300, 350}));
}

TEST(dynamic_fws_multimap_test, int_resize_1) {
  auto mm = build_test_map_1();

  mm[0].resize(4);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(14, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8, 0, 0}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));

  mm[1].resize(3);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(13, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8, 0, 0}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));

  mm[1].resize(6, 123);

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(16, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8, 0, 0}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 123, 123, 123}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));
}

TEST(dynamic_fws_multimap_test, pop_back_1) {
  auto mm = build_test_map_1();

  mm[2].pop_back();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(11, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350}));

  mm[1].pop_back();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(10, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350}));

  mm[0].pop_back();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(9, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350}));

  mm[0].pop_back();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], IsEmpty());
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350}));
}

TEST(dynamic_fws_multimap_test, clear_1) {
  auto mm = build_test_map_1();

  mm[0].clear();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(10, mm.element_count());
  EXPECT_THAT(mm[0], IsEmpty());
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));
}

TEST(dynamic_fws_multimap_test, clear_2) {
  auto mm = build_test_map_1();

  mm[1].clear();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(8, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], IsEmpty());
  EXPECT_THAT(mm[2], ElementsAreArray({100, 200, 250, 300, 350, 400}));
}

TEST(dynamic_fws_multimap_test, clear_3) {
  auto mm = build_test_map_1();

  mm[2].clear();

  ASSERT_EQ(3, mm.index_size());
  EXPECT_EQ(6, mm.element_count());
  EXPECT_THAT(mm[0], ElementsAreArray({4, 8}));
  EXPECT_THAT(mm[1], ElementsAreArray({15, 16, 23, 42}));
  EXPECT_THAT(mm[2], IsEmpty());
}

namespace {

// -> dynamic_fws_multimap::get_order

#ifdef MOTIS_AVX2
template <typename T>
T get_order_bmi1(T const size) {
  if constexpr (sizeof(T) == 8) {
    return _tzcnt_u64(size);
  } else {
    return _tzcnt_u32(static_cast<std::uint32_t>(size));
  }
}
#endif

template <typename T>
T get_order_loop(T const size) {
  for (auto order = T{0}, value = T{1}; order < sizeof(T) * 8;
       ++order, value = value << 1) {
    if (value == size) {
      return order;
    }
  }
  throw utl::fail("dynamic_fws_multimap::get_order: not found for {}", size);
}

}  // namespace

TEST(dynamic_fws_multimap_test, get_order_loop) {
  for (std::uint16_t i = 0U; i < 16; ++i) {
    EXPECT_EQ(get_order_loop(static_cast<std::uint16_t>(1U) << i), i);
  }

  for (std::uint32_t i = 0U; i < 32; ++i) {
    EXPECT_EQ(get_order_loop(1U << i), i);
  }

  for (std::uint64_t i = 0ULL; i < 64; ++i) {
    EXPECT_EQ(get_order_loop(1ULL << i), i);
  }
}

#ifdef MOTIS_AVX2
TEST(dynamic_fws_multimap_test, get_order_bmi1) {
  for (std::uint16_t i = 0U; i < 16; ++i) {
    EXPECT_EQ(get_order_bmi1(static_cast<std::uint16_t>(1U) << i), i);
  }

  for (std::uint32_t i = 0U; i < 32; ++i) {
    EXPECT_EQ(get_order_bmi1(1U << i), i);
  }

  for (std::uint64_t i = 0ULL; i < 64; ++i) {
    EXPECT_EQ(get_order_bmi1(1ULL << i), i);
  }
}
#endif

}  // namespace motis
