#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "motis/gbfs/lru_cache.h"

using namespace motis::gbfs;

TEST(gbfs_lru_cache, get_or_compute_propagates_exceptions) {
  auto cache = lru_cache<int, int>{4U};

  EXPECT_THROW(
      {
        cache.get_or_compute(
            1, []() -> std::shared_ptr<int> { throw std::runtime_error{"x"}; });
      },
      std::runtime_error);

  // a failed computation must not be remembered
  EXPECT_FALSE(cache.contains(1));

  auto const value =
      cache.get_or_compute(1, []() { return std::make_shared<int>(42); });
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 42);
  EXPECT_TRUE(cache.contains(1));
}

TEST(gbfs_lru_cache, evicts_least_recently_used) {
  auto cache = lru_cache<int, int>{2U};

  auto const put = [&](int const k) {
    return *cache.get_or_compute(k, [&]() { return std::make_shared<int>(k); });
  };

  put(1);
  put(2);
  put(1);  // 2 is now the least recently used one
  put(3);

  EXPECT_EQ(cache.size(), 2U);
  EXPECT_TRUE(cache.contains(1));
  EXPECT_FALSE(cache.contains(2));
  EXPECT_TRUE(cache.contains(3));
}

TEST(gbfs_lru_cache, concurrent_hits_keep_the_order_consistent) {
  auto cache = lru_cache<int, int>{8U};
  for (auto i = 0; i != 8; ++i) {
    cache.get_or_compute(i, [&]() { return std::make_shared<int>(i); });
  }

  // hits used to reorder lru_order_ under a shared lock
  {
    auto threads = std::vector<std::jthread>{};
    for (auto t = 0; t != 4; ++t) {
      threads.emplace_back([&]() {
        for (auto r = 0; r != 2000; ++r) {
          for (auto i = 0; i != 8; ++i) {
            EXPECT_EQ(*cache.get_or_compute(
                          i, [&]() { return std::make_shared<int>(-1); }),
                      i);
          }
        }
      });
    }
  }

  EXPECT_EQ(cache.size(), 8U);
  for (auto i = 0; i != 8; ++i) {
    EXPECT_TRUE(cache.contains(i));
  }
}
