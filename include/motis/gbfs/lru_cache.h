#pragma once

#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "motis/types.h"

#include "utl/helpers/algorithm.h"

namespace motis::gbfs {

template <typename Key, typename Value>
class lru_cache {
public:
  explicit lru_cache(std::size_t const max_size) : max_size_{max_size} {}

  lru_cache(lru_cache const& o) : max_size_{o.max_size_} {
    auto read_lock = std::shared_lock{o.mutex_};
    cache_map_ = o.cache_map_;
    lru_order_ = o.lru_order_;
    pending_computations_.clear();
  }

  lru_cache& operator=(lru_cache const& o) {
    if (this != &o) {
      auto read_lock = std::shared_lock{o.mutex_};
      auto write_lock = std::unique_lock{mutex_};
      max_size_ = o.max_size_;
      cache_map_ = o.cache_map_;
      lru_order_ = o.lru_order_;
      pending_computations_.clear();
    }
    return *this;
  }

  template <typename F>
  std::shared_ptr<Value> get_or_compute(Key const key, F compute_fn) {
    // check with shared lock if entry already exists
    {
      auto read_lock = std::shared_lock{mutex_};
      if (auto const it = cache_map_.find(key); it != cache_map_.end()) {
        auto value = it->second->value_;
        read_lock.unlock();
        touch(key);
        return value;
      }
    }

    // not found -> acquire exclusive lock to modify the cache
    auto write_lock = std::unique_lock{mutex_};

    // check again in case another thread inserted it
    if (auto it = cache_map_.find(key); it != cache_map_.end()) {
      move_to_front(key);
      return it->second->value_;
    }

    // if another thread is already computing it, wait for it
    if (auto it = pending_computations_.find(key);
        it != pending_computations_.end()) {
      auto future = it->second;
      write_lock.unlock();
      return future.get();
    }

    // create pending computation
    auto promise = std::promise<std::shared_ptr<Value>>{};
    auto shared_future = promise.get_future().share();
    pending_computations_[key] = shared_future;
    write_lock.unlock();

    // compute the value
    auto value = std::shared_ptr<Value>{};
    try {
      value = compute_fn();
    } catch (...) {
      auto const ex = std::current_exception();
      write_lock.lock();
      pending_computations_.erase(key);
      write_lock.unlock();
      promise.set_exception(ex);
      throw;
    }

    // store the result
    write_lock.lock();

    if (lru_order_.size() >= max_size_) {
      // evict least recently used cache entry
      auto const last_key = lru_order_.back();
      cache_map_.erase(last_key);
      lru_order_.pop_back();
    }

    cache_map_.try_emplace(
        key, std::make_shared<cache_entry>(cache_entry{key, value}));
    lru_order_.insert(lru_order_.begin(), key);
    pending_computations_.erase(key);
    promise.set_value(value);

    return value;
  }

  std::shared_ptr<Value> get(Key const key) const {
    auto read_lock = std::shared_lock{mutex_};
    if (auto it = cache_map_.find(key); it != cache_map_.end()) {
      return it->second->value_;
    }
    return nullptr;
  }

  bool contains(Key const key) const {
    auto read_lock = std::shared_lock{mutex_};
    return cache_map_.find(key) != cache_map_.end();
  }

  /// adds an entry to the cache if there is still space or updates
  /// an existing entry if it already exists
  template <typename F>
  bool try_add_or_update(Key const key, F compute_fn) {
    auto write_lock = std::unique_lock{mutex_};

    if (auto it = cache_map_.find(key); it != cache_map_.end()) {
      it->second =
          std::make_shared<cache_entry>(cache_entry{key, compute_fn()});
      move_to_front(key);
      return true;
    }

    if (lru_order_.size() >= max_size_) {
      return false;
    }

    cache_map_.try_emplace(
        key, std::make_shared<cache_entry>(cache_entry{key, compute_fn()}));
    lru_order_.insert(lru_order_.begin(), key);
    return true;
  }

  void remove(Key const key) {
    auto write_lock = std::unique_lock{mutex_};
    if (auto it = cache_map_.find(key); it != cache_map_.end()) {
      if (auto const lru_it = utl::find(lru_order_, key);
          lru_it != lru_order_.end()) {
        lru_order_.erase(lru_it);
      }
      cache_map_.erase(it);
    }
  }

  std::size_t size() const {
    auto read_lock = std::shared_lock{mutex_};
    return lru_order_.size();
  }

  bool empty() const {
    auto read_lock = std::shared_lock{mutex_};
    return lru_order_.empty();
  }

private:
  struct cache_entry {
    Key key_{};
    std::shared_ptr<Value> value_{};
  };

  // runs on every cache hit: skip instead of waiting for the lock
  void touch(Key const key) {
    if (auto write_lock = std::unique_lock{mutex_, std::try_to_lock};
        write_lock.owns_lock()) {
      move_to_front(key);
    }
  }

  void move_to_front(Key const key) {
    auto const it = utl::find(lru_order_, key);
    if (it != lru_order_.end()) {
      lru_order_.erase(it);
      lru_order_.insert(lru_order_.begin(), key);
    }
  }

  std::size_t max_size_;
  hash_map<Key, std::shared_ptr<cache_entry>> cache_map_;
  std::vector<Key> lru_order_;
  hash_map<Key, std::shared_future<std::shared_ptr<Value>>>
      pending_computations_{};
  mutable std::shared_mutex mutex_;
};

}  // namespace motis::gbfs
