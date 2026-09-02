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
    copy_entries(o);
  }

  lru_cache& operator=(lru_cache const& o) {
    if (this != &o) {
      // both exclusively, in a deadlock free order: a = b and b = a running
      // concurrently would otherwise take the two mutexes in opposite order
      auto locks = std::scoped_lock{mutex_, o.mutex_};
      max_size_ = o.max_size_;
      copy_entries(o);
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
      // hand the error to everyone waiting on this key and drop the pending
      // computation - otherwise the promise dies unfulfilled and every waiter
      // (now and forever, as the entry would never be erased) sees a
      // broken_promise instead of the actual error
      auto const ex = std::current_exception();
      write_lock.lock();
      pending_computations_.erase(key);
      write_lock.unlock();
      promise.set_exception(ex);
      throw;
    }

    // store the result
    write_lock.lock();

    // max_size_ == 0 disables the cache - nothing to evict, nothing to store
    if (max_size_ != 0U) {
      if (lru_order_.size() >= max_size_) {
        // evict least recently used cache entry
        auto const last_key = lru_order_.back();
        cache_map_.erase(last_key);
        lru_order_.pop_back();
      }

      // another thread may have inserted the key meanwhile (try_add_or_update
      // does not go through pending_computations_) - keep its entry, but do
      // not add a second lru_order_ entry for the same key
      if (cache_map_
              .try_emplace(
                  key, std::make_shared<cache_entry>(cache_entry{key, value}))
              .second) {
        lru_order_.insert(lru_order_.begin(), key);
      } else {
        move_to_front(key);
      }
    }
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

  template <typename F>
  void update_if_exists(Key const key, F update_fn) {
    auto write_lock = std::unique_lock{mutex_};
    if (auto it = cache_map_.find(key); it != cache_map_.end()) {
      it->second->value_ = update_fn(it->second->value_);
      move_to_front(key);
    }
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

    if (max_size_ == 0U || lru_order_.size() >= max_size_) {
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

  std::vector<std::pair<Key, std::shared_ptr<Value>>> get_all_entries() const {
    auto read_lock = std::shared_lock{mutex_};
    auto entries = std::vector<std::pair<Key, std::shared_ptr<Value>>>{};
    entries.reserve(lru_order_.size());
    for (auto it = lru_order_.rbegin(); it != lru_order_.rend(); ++it) {
      if (auto const map_it = cache_map_.find(*it);
          map_it != cache_map_.end()) {
        entries.emplace_back(map_it->first, map_it->second->value_);
      }
    }
    return entries;
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

  /// copies o's entries into this cache. the entries themselves are copied,
  /// not shared: a cache is copied to hand it to the next gbfs_data, which
  /// then writes to it (try_add_or_update) while requests may still be reading
  /// from the previous one.
  /// the values are immutable once built, so those stay shared.
  /// expects o's mutex to be held (and this cache's, if it is reachable).
  void copy_entries(lru_cache const& o) {
    cache_map_.clear();
    cache_map_.reserve(o.cache_map_.size());
    for (auto const& [key, entry] : o.cache_map_) {
      cache_map_.emplace(key, std::make_shared<cache_entry>(*entry));
    }
    lru_order_ = o.lru_order_;
    pending_computations_.clear();
  }

  /// moves key to the front of the lru order. only a hint for the eviction
  /// order, so it is skipped instead of waited for when the cache is busy -
  /// this runs on every cache hit and must not serialize them.
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
