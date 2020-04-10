#pragma once

#include <cassert>
#include <algorithm>
#include <vector>

namespace motis {

template <typename T, std::size_t MaxBucket,
          typename GetBucketFn  // GetBucketFn(T) -> size_t <= MaxBucket
          >
class dial {
public:
  explicit dial(GetBucketFn get_bucket = GetBucketFn())
      : get_bucket_(std::forward<GetBucketFn>(get_bucket)),
        current_bucket_(0),
        size_(0),
        buckets_(MaxBucket + 1) {}

  template <typename El>
  inline void push(El&& el) {
    auto const dist = get_bucket_(el);
    assert(dist <= MaxBucket);

    buckets_[dist].emplace_back(std::forward<El>(el));
    current_bucket_ = std::min(current_bucket_, dist);
    ++size_;
  }

  inline T const& top() {
    assert(!empty());
    current_bucket_ = get_next_bucket();
    assert(!buckets_[current_bucket_].empty());
    return buckets_[current_bucket_].back();
  }

  inline void pop() {
    assert(!empty());
    current_bucket_ = get_next_bucket();
    buckets_[current_bucket_].pop_back();
    --size_;
  }

  inline std::size_t size() const { return size_; }

  inline bool empty() const { return size_ == 0; }

private:
  inline std::size_t get_next_bucket() const {
    assert(size_ != 0);
    auto bucket = current_bucket_;
    while (bucket < buckets_.size() && buckets_[bucket].empty()) {
      ++bucket;
    }
    return bucket;
  }

  GetBucketFn get_bucket_;
  std::size_t current_bucket_;
  std::size_t size_;
  std::vector<std::vector<T>> buckets_;
};

}  // namespace motis
