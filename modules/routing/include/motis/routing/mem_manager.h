#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>

#include "motis/routing/allocator.h"

namespace motis::routing {

struct mem_manager {
public:
  explicit mem_manager(std::size_t const initial_size)
      : allocations_(0), alloc_(initial_size) {}

  mem_manager(mem_manager const&) = delete;
  mem_manager& operator=(mem_manager const&) = delete;

  mem_manager(mem_manager&&) = delete;
  mem_manager& operator=(mem_manager&&) = delete;

  ~mem_manager() = default;

  void reset() {
    allocations_ = 0;
    alloc_.clear();
    for (auto& labels : node_labels_) {
      labels.clear();
    }
  }

  template <typename T, typename... Args>
  T* create(Args&&... args) {
    ++allocations_;
    return new (alloc_.alloc(sizeof(T)))  // NOLINT
        T(std::forward<Args>(args)...);
  }

  template <typename T>
  void release(T* ptr) {
    alloc_.dealloc(ptr);
  }

  template <typename T>
  std::vector<std::vector<T*>>* get_node_labels(std::size_t size) {
    node_labels_.resize(size);
    return reinterpret_cast<std::vector<std::vector<T*>>*>(&node_labels_);
  }

  size_t allocations() const { return allocations_; }

  size_t get_num_bytes_in_use() const { return alloc_.get_num_bytes_in_use(); }

private:
  size_t allocations_;
  allocator alloc_;
  std::vector<std::vector<void*>> node_labels_;
};

}  // namespace motis::routing
