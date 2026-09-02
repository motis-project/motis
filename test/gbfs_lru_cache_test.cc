#include "gtest/gtest.h"

#include <barrier>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
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

TEST(gbfs_lru_cache, get_or_compute_forwards_exceptions_to_waiters) {
  auto cache = lru_cache<int, int>{4U};

  // both threads enter get_or_compute for the same key: one computes and
  // throws, the other waits on the pending computation and has to see the
  // same error instead of a broken promise
  auto entered = std::barrier{2};
  auto const run = [&]() {
    entered.arrive_and_wait();
    try {
      cache.get_or_compute(1, []() -> std::shared_ptr<int> {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        throw std::runtime_error{"boom"};
      });
    } catch (std::exception const& e) {
      return std::string{e.what()};
    }
    return std::string{"no exception"};
  };

  auto a = std::string{};
  auto b = std::string{};
  {
    auto t0 = std::jthread{[&]() { a = run(); }};
    auto t1 = std::jthread{[&]() { b = run(); }};
  }

  EXPECT_EQ(a, "boom");
  EXPECT_EQ(b, "boom");
  EXPECT_FALSE(cache.contains(1));

  // the key is usable again afterwards
  EXPECT_EQ(*cache.get_or_compute(1, []() { return std::make_shared<int>(7); }),
            7);
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

TEST(gbfs_lru_cache, zero_max_size_computes_every_time) {
  auto cache = lru_cache<int, int>{0U};

  auto calls = 0;
  auto const put = [&]() {
    return cache.get_or_compute(1, [&]() {
      ++calls;
      return std::make_shared<int>(1);
    });
  };

  EXPECT_EQ(*put(), 1);
  EXPECT_EQ(*put(), 1);
  EXPECT_EQ(calls, 2);
  EXPECT_TRUE(cache.empty());
  EXPECT_FALSE(
      cache.try_add_or_update(1, []() { return std::make_shared<int>(1); }));
}

TEST(gbfs_lru_cache, copy_does_not_share_entries) {
  auto cache = lru_cache<int, int>{4U};
  cache.get_or_compute(1, []() { return std::make_shared<int>(1); });

  // the cache is copied to hand it to the next gbfs_data, which then writes to
  // it while requests may still be reading from the previous one - so writing
  // to the copy must not touch the original's entry
  auto copy = cache;
  ASSERT_TRUE(
      copy.try_add_or_update(1, []() { return std::make_shared<int>(2); }));

  EXPECT_EQ(*cache.get(1), 1);
  EXPECT_EQ(*copy.get(1), 2);
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
