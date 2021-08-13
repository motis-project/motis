#pragma once

#include <cstddef>
#include <vector>

#include "utl/parser/buffer.h"

namespace motis::paxmon {

template <typename T>
struct allocator {
  static_assert(sizeof(T) >= sizeof(void*));

  allocator() = default;

  ~allocator() = default;

  allocator(allocator const&) = delete;
  allocator& operator=(allocator const&) = delete;

  allocator(allocator&&) = delete;
  allocator& operator=(allocator&&) = delete;

  template <typename... Args>
  inline T* create(Args&&... args) {
    return new (alloc()) T(std::forward<Args>(args)...);  // NOLINT
  }

  inline void release(T* ptr) {
    ptr->~T();
    dealloc(ptr);
  }

  inline std::size_t elements_allocated() const { return elements_allocated_; }
  inline std::size_t bytes_allocated() const { return bytes_allocated_; }
  inline std::size_t free_list_size() const { return free_list_size_; }
  inline std::size_t allocation_count() const { return allocation_count_; }
  inline std::size_t release_count() const { return release_count_; }

private:
  inline void* alloc() {
    ++elements_allocated_;
    ++allocation_count_;
    if (free_list_.next_ != nullptr) {
      --free_list_size_;
      return free_list_.take();
    }
    if (next_ptr_ == nullptr || next_ptr_ + sizeof(T) >= end_ptr_) {
      auto& new_block = blocks_.emplace_back(utl::buffer{next_block_size()});
      next_ptr_ = new_block.begin();
      end_ptr_ = new_block.end();
      bytes_allocated_ += new_block.size();
    }
    auto const mem_ptr = next_ptr_;
    next_ptr_ += sizeof(T);
    return mem_ptr;
  }

  inline void dealloc(void* ptr) {
    --elements_allocated_;
    ++release_count_;
    ++free_list_size_;
    free_list_.push(ptr);
  }

  inline std::size_t next_block_size() {
    if (blocks_.empty()) {
      return 10'000 * sizeof(T);
    } else {
      return 100'000 * sizeof(T);
    }
  }

  std::vector<utl::buffer> blocks_;
  unsigned char* next_ptr_{nullptr};
  unsigned char* end_ptr_{nullptr};
  std::size_t elements_allocated_{};
  std::size_t bytes_allocated_{};
  std::size_t free_list_size_{};
  std::size_t allocation_count_{};
  std::size_t release_count_{};

  struct node {
    inline void* take() {
      auto const ptr = next_;
      next_ = next_->next_;
      return ptr;
    }
    inline void push(void* p) {
      auto const mem_ptr = reinterpret_cast<node*>(p);
      mem_ptr->next_ = next_;
      next_ = mem_ptr;
    }
    node* next_{nullptr};
  } free_list_;
};

}  // namespace motis::paxmon
