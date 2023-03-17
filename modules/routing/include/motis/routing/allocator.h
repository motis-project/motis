#pragma once

#include <numeric>
#include <vector>

#include "fmt/core.h"

#include "utl/parser/buffer.h"

namespace motis::routing {

constexpr auto const alloc_tracing = false;

template <typename... Ts>
void alloc_trace(Ts&&... args) {
  if (alloc_tracing) {
    fmt::print(std::cerr, std::forward<Ts>(args)...);
  }
}

struct allocator {
  explicit allocator(size_t const initial_size) {
    mem_.emplace_back(initial_size);
    clear();
  }

  ~allocator() = default;

  allocator(allocator const&) = delete;
  allocator& operator=(allocator const&) = delete;

  allocator(allocator&&) = delete;
  allocator& operator=(allocator&&) = delete;

  inline void dealloc(void* p) {
    alloc_trace("dealloc {}\n", fmt::ptr(p));
    list_.push(p);
  }

  inline void* alloc(size_t const size) {
    alloc_trace("alloc size={} -> ", size);
    if (list_.next_ != nullptr) {
      auto const mem_ptr = list_.take();
      alloc_trace(" freelist -> {}\n", fmt::ptr(mem_ptr));
      return mem_ptr;
    } else if (next_ptr_ + size < end_ptr_) {
      auto const mem_ptr = next_ptr_;
      next_ptr_ += size;
      alloc_trace(" membuf -> {}\n", fmt::ptr(mem_ptr));
      return mem_ptr;
    } else {
      mem_.emplace_back(mem_.back().size_ * 2);
      set_range();
      assert(next_ptr_ + size < end_ptr_);
      auto const mem_ptr = next_ptr_;
      next_ptr_ += size;
      alloc_trace(" new membuf -> {}\n", fmt::ptr(mem_ptr));
      return mem_ptr;
    }
  }

  inline void clear() {
    list_.next_ = nullptr;
    mem_.resize(1);
    set_range();
  }

  inline size_t get_num_bytes_in_use() const {
    return std::accumulate(begin(mem_), end(mem_), size_t{0U},
                           [](size_t const sum, utl::buffer const& buf) {
                             return sum + buf.size_;
                           });
  }

private:
  inline void set_range() {
    next_ptr_ = mem_.back().begin();
    end_ptr_ = mem_.back().end();
  }

  std::vector<utl::buffer> mem_;
  unsigned char* next_ptr_{nullptr};
  unsigned char* end_ptr_{nullptr};

  struct node {
    inline void* take() {
      assert(next_ != nullptr);
      auto const ptr = next_;
      alloc_trace("  take: next = {}  ->   {}\n", fmt::ptr(next_),
                  fmt::ptr(next_->next_));
      next_ = next_->next_;
      return ptr;
    }
    inline void push(void* p) {
      auto const mem_ptr = reinterpret_cast<node*>(p);
      mem_ptr->next_ = next_;
      alloc_trace("  push: next = {}  ->  {}\n", fmt::ptr(next_),
                  fmt::ptr(mem_ptr));
      next_ = mem_ptr;
    }
    node* next_{nullptr};
  } list_{};
};

}  // namespace motis::routing
