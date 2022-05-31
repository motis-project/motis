#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <bitset>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "cista/reflection/comparable.h"

namespace motis::paxmon {

template <typename Type>
struct allocator {
  static constexpr auto const INITIAL_BLOCK_SIZE = 10'000;
  static constexpr auto const ADDITIONAL_BLOCK_SIZE = 100'000;
  static constexpr auto const MAX_BLOCK_SIZE =
      std::max(INITIAL_BLOCK_SIZE, ADDITIONAL_BLOCK_SIZE);

  struct block {
    block() = default;
    explicit block(std::size_t size) : ptr_{operator new(size)}, size_{size} {}

    ~block() {
      if constexpr (!std::is_trivially_destructible_v<Type>) {
        for (auto i = 0ULL; i < size_ / sizeof(Type); ++i) {
          if (in_use_[i]) {
            get(i)->~Type();
          }
        }
      }
      operator delete(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }

    block(block const& o)
        : ptr_{operator new(o.size_)}, size_{o.size_}, in_use_{o.in_use_} {
      if constexpr (std::is_trivially_copyable_v<Type>) {
        std::memcpy(ptr_, o.ptr_, size_);
      } else {
        for (auto i = 0ULL; i < size_ / sizeof(Type); ++i) {
          if (o.in_use_[i]) {
            new (get(i)) Type{*o.get(i)};
          }
        }
      }
    }

    block(block&& o) noexcept : block{} { swap(*this, o); }

    block& operator=(block const o) {
      swap(*this, o);
      return *this;
    }

    block& operator=(block&& o) noexcept {
      swap(*this, o);
      return *this;
    }

    friend void swap(block& a, block& b) noexcept {
      using std::swap;
      swap(a.ptr_, b.ptr_);
      swap(a.size_, b.size_);
      swap(a.in_use_, b.in_use_);
    }

    inline std::size_t size() const { return size_; }
    inline void* data() const { return ptr_; }

    inline Type* get(std::size_t const index) const {
      return reinterpret_cast<Type*>(  // NOLINT(performance-no-int-to-ptr)
          reinterpret_cast<std::uintptr_t>(ptr_) +
          static_cast<std::uintptr_t>(index * sizeof(Type)));
    }

    void* ptr_{};
    std::size_t size_{};
    std::bitset<MAX_BLOCK_SIZE> in_use_;
  };

  struct pointer {
    static constexpr auto const INVALID_BLOCK =
        std::numeric_limits<std::uint32_t>::max();
    static constexpr auto const INVALID_OFFSET =
        std::numeric_limits<std::uint32_t>::max();

    CISTA_COMPARABLE()

    explicit operator bool() const noexcept {
      return block_index_ != INVALID_BLOCK;
    }

    std::uint32_t block_index_{INVALID_BLOCK};
    std::uint32_t block_offset_{INVALID_OFFSET};
  };

  static_assert(sizeof(Type) >= sizeof(pointer));
  static_assert(std::max(INITIAL_BLOCK_SIZE, ADDITIONAL_BLOCK_SIZE) *
                    sizeof(Type) <=
                std::numeric_limits<std::uint32_t>::max());

  template <typename... Args>
  inline std::pair<pointer, Type*> create(Args&&... args) {
    auto const ptr = alloc();
    auto* mem_ptr = get(ptr);
    new (mem_ptr) Type(std::forward<Args>(args)...);  // NOLINT
    return {ptr, mem_ptr};
  }

  inline void release(pointer ptr) {
    get(ptr)->~Type();
    dealloc(ptr);
  }

  inline Type* get(pointer const ptr) const {
    return reinterpret_cast<Type*>(  //  NOLINT
        reinterpret_cast<std::uintptr_t>(blocks_[ptr.block_index_].data()) +
        static_cast<std::uintptr_t>(ptr.block_offset_));
  }

  inline Type* get_checked(pointer const ptr) const {
    if (!ptr) {
      return nullptr;
    }
    if (ptr.block_index_ >= blocks_.size() ||
        ptr.block_index_ >= blocks_[ptr.block_index_].size()) {
      throw std::out_of_range{
          "motis::paxmon::allocator::get_checked: invalid pointer"};
    }
    return get(ptr);
  }

  inline std::size_t elements_allocated() const { return elements_allocated_; }
  inline std::size_t bytes_allocated() const { return bytes_allocated_; }
  inline std::size_t free_list_size() const { return free_list_size_; }
  inline std::size_t allocation_count() const { return allocation_count_; }
  inline std::size_t release_count() const { return release_count_; }

private:
  inline pointer alloc() {
    ++elements_allocated_;
    ++allocation_count_;
    if (free_list_.next_) {
      --free_list_size_;
      auto const ptr = free_list_.take(*this);
      mark_in_use(ptr);
      return ptr;
    }
    if (!next_ptr_ ||
        end_ptr_.block_offset_ - next_ptr_.block_offset_ < sizeof(Type)) {
      auto& new_block = blocks_.emplace_back(block{next_block_size()});
      auto const block_index = static_cast<std::uint32_t>(blocks_.size() - 1);
      next_ptr_ = {block_index, 0};
      end_ptr_ = {block_index, static_cast<std::uint32_t>(new_block.size())};
      bytes_allocated_ += new_block.size();
    }
    auto const ptr = next_ptr_;
    mark_in_use(ptr);
    next_ptr_.block_offset_ += sizeof(Type);
    return ptr;
  }

  inline void dealloc(pointer ptr) {
    --elements_allocated_;
    ++release_count_;
    ++free_list_size_;
    mark_free(ptr);
    free_list_.push(*this, ptr);
  }

  inline std::size_t next_block_size() {
    if (blocks_.empty()) {
      return INITIAL_BLOCK_SIZE * sizeof(Type);
    } else {
      return ADDITIONAL_BLOCK_SIZE * sizeof(Type);
    }
  }

  inline void mark_in_use(pointer ptr) {
    blocks_[ptr.block_index_].in_use_.set(ptr.block_offset_ / sizeof(Type));
  }

  inline void mark_free(pointer ptr) {
    blocks_[ptr.block_index_].in_use_.reset(ptr.block_offset_ / sizeof(Type));
  }

  std::vector<block> blocks_;
  pointer next_ptr_{};
  pointer end_ptr_{};

  std::size_t elements_allocated_{};
  std::size_t bytes_allocated_{};
  std::size_t free_list_size_{};
  std::size_t allocation_count_{};
  std::size_t release_count_{};

  struct node {
    inline pointer take(allocator const& a) {
      auto const ptr = next_;
      next_ = reinterpret_cast<node*>(a.get(next_))->next_;
      return ptr;
    }
    inline void push(allocator const& a, pointer ptr) {
      auto const mem_ptr = reinterpret_cast<node*>(a.get(ptr));
      mem_ptr->next_ = next_;
      next_ = ptr;
    }
    pointer next_{};
  } free_list_;
};

}  // namespace motis::paxmon
