#pragma once

#include "motis/vector.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace motis {

template <typename T, typename SizeType = std::uint32_t>
struct dynamic_fws_multimap {
  using entry_type = T;
  using size_type = SizeType;

  struct index_type {
    size_type begin_{};
    size_type size_{};
    size_type capacity_{};
  };

  struct bucket {
    friend dynamic_fws_multimap;

    using size_type = size_type;
    using value_type = T;
    using iterator = typename mcd::vector<T>::iterator;
    using const_iterator = typename mcd::vector<T>::const_iterator;

    size_type index() const { return index_; }
    size_type size() const { return get_index().size_; }
    size_type capacity() const { return get_index().capacity_; }
    [[nodiscard]] bool empty() const { return size() == 0; }

    iterator begin() const {
      return multimap_.data_.begin() + get_index().begin_;
    }

    iterator end() const {
      auto const& index = get_index();
      return multimap_.data_.begin() + index.begin_ + index.size_;
    }

    iterator cbegin() const { return begin(); }
    iterator cend() const { return cend(); }

    friend iterator begin(bucket const& b) { return b.begin(); }
    friend iterator end(bucket const& b) { return b.end(); }

    T& operator[](size_type index) {
      return multimap_.data_[get_index().begin_ + index];
    }

    T& operator[](size_type index) const {
      return multimap_.data_[get_index().begin_ + index];
    }

    T& at(size_type index) const {
      auto const& idx = get_index();
      if (index >= idx.size_) {
        throw std::out_of_range{
            "dynamic_fws_multimap::bucket::at() out of range"};
      }
      return multimap_.data_[idx.begin_ + index];
    }

    size_type push_back(entry_type const& val) {
      return multimap_.push_back_entry(index_, val);
    }

    template <typename... Args>
    size_type emplace_back(Args&&... args) {
      return multimap_.emplace_back_entry(index_, std::forward<Args>(args)...);
    }

    void reserve(size_type new_size) {
      if (new_size > capacity()) {
        multimap_.grow_bucket(index_, get_index(), new_size);
      }
    }

  protected:
    bucket(dynamic_fws_multimap& multimap, size_type index)
        : multimap_(multimap), index_(index) {}

    index_type& get_index() const { return multimap_.index_[index_]; }

    dynamic_fws_multimap& multimap_;
    size_type const index_;
  };

  struct iterator {
    friend dynamic_fws_multimap;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = bucket;
    using difference_type = int;
    using pointer = value_type;
    using reference = value_type;

    value_type operator*() const { return multimap_.at(index_); }
    value_type operator->() const { return multimap_.at(index_); }

    iterator& operator+=(difference_type n) {
      index_ += n;
      return *this;
    }

    iterator& operator-=(difference_type n) {
      index_ -= n;
      return *this;
    }

    iterator& operator++() {
      ++index_;
      return *this;
    }

    iterator& operator--() {
      ++index_;
      return *this;
    }

    iterator operator+(difference_type n) const {
      return {multimap_, index_ + n};
    }

    iterator operator-(difference_type n) const {
      return {multimap_, index_ - n};
    }

    int operator-(iterator const& rhs) const { return index_ - rhs.index_; };

    value_type operator[](difference_type n) {
      return multimap_.at(index_ + n);
    }

    bool operator<(iterator const& rhs) const { return index_ < rhs.index_; }
    bool operator<=(iterator const& rhs) const { return index_ <= rhs.index_; }
    bool operator>(iterator const& rhs) const { return index_ > rhs.index_; }
    bool operator>=(iterator const& rhs) const { return index_ >= rhs.index_; }

    bool operator==(iterator const& rhs) const {
      return &multimap_ == &rhs.multimap_ && index_ == rhs.index_;
    }

    bool operator!=(iterator const& rhs) const {
      return &multimap_ != &rhs.multimap_ || index_ != rhs.index_;
    }

  protected:
    iterator(dynamic_fws_multimap& multimap, size_type index)
        : multimap_(multimap), index_(index) {}

    dynamic_fws_multimap& multimap_;
    size_type index_;
  };

  bucket operator[](size_type index) {
    if (index >= index_.size()) {
      index_.resize(index + 1);
    }
    return {*this, index};
  }

  bucket at(size_type index) {
    if (index >= index_.size()) {
      throw std::out_of_range{"dynamic_fws_multimap::at() out of range"};
    } else {
      return {*this, index};
    }
  }

  bucket emplace_back() { return this[index_size()]; }

  size_type index_size() const { return index_.size(); }
  size_type data_size() const { return index_.size(); }
  size_type element_count() const { return element_count_; }
  [[nodiscard]] bool empty() const { return index_size() == 0; }

  iterator begin() { return {*this, 0}; }
  iterator end() { return iterator{*this, index_.size()}; }

  friend iterator begin(dynamic_fws_multimap const& m) { return m.begin(); }
  friend iterator end(dynamic_fws_multimap const& m) { return m.end(); }

  mcd::vector<T>& data() { return data_; }

protected:
  static constexpr auto const INITIAL_CAPACITY = 2;
  static constexpr auto const GROW_FACTOR = 2;

  size_type insert_new_entry(size_type const map_index) {
    assert(map_index < index_.size());
    auto& idx = index_[map_index];
    if (idx.size_ == idx.capacity_) {
      grow_bucket(map_index, idx);
    }
    auto const data_index = idx.begin_ + idx.size_;
    idx.size_++;
    assert(idx.size_ <= idx.capacity_);
    return data_index;
  }

  void grow_bucket(size_type const map_index, index_type& idx) {
    auto const new_capacity =
        idx.capacity_ == 0 ? INITIAL_CAPACITY : idx.capacity_ * GROW_FACTOR;
    grow_bucket(map_index, idx, new_capacity);
  }

  void grow_bucket(size_type const map_index, index_type& idx,
                   size_type const new_capacity) {
    if (idx.capacity_ == 0) {
      // new bucket
      idx.begin_ = data_.size();
      data_.resize(data_.size() + new_capacity);
      idx.capacity_ = new_capacity;
    } else if (idx.begin_ + idx.capacity_ == data_.size()) {
      // last bucket
      auto const additional_capacity = new_capacity - idx.capacity_;
      data_.resize(data_.size() + additional_capacity);
      idx.capacity_ = new_capacity;
    } else {
      // move to end of data vector
      auto const new_begin = data_.size();
      data_.resize(data_.size() + new_capacity);
      move_entries(map_index, idx.begin_, new_begin, idx.size_);
      idx.begin_ = new_begin;
      idx.capacity_ = new_capacity;
    }
  }

  virtual void move_entries(size_type const /*map_index*/,
                            size_type const old_data_index,
                            size_type const new_data_index,
                            size_type const count) {
    auto old_data = &data_[old_data_index];
    auto new_data = &data_[new_data_index];
    for (auto i = static_cast<size_type>(0); i < count;
         ++i, ++old_data, ++new_data) {
      new (new_data) T(std::move(*old_data));
      old_data->~T();
    }
  }

  size_type push_back_entry(size_type const map_index, entry_type const& val) {
    auto const data_index = insert_new_entry(map_index);
    data_[data_index] = val;
    ++element_count_;
    return data_index;
  }

  template <typename... Args>
  size_type emplace_back_entry(size_type const map_index, Args&&... args) {
    auto const data_index = insert_new_entry(map_index);
    new (&data_[data_index]) T{std::forward<Args>(args)...};
    ++element_count_;
    return data_index;
  }

public:
  mcd::vector<index_type> index_;
  mcd::vector<T> data_;
  size_type element_count_{};
};

}  // namespace motis
