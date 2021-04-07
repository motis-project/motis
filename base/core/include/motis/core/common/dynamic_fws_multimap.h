#pragma once

#include "motis/vector.h"

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace motis {

template <typename Derived, typename T, typename SizeType = std::uint32_t>
struct dynamic_fws_multimap_base {
  using entry_type = T;
  using size_type = SizeType;

  struct index_type {
    size_type begin_{};
    size_type size_{};
    size_type capacity_{};
  };

  template <bool Const>
  struct bucket {
    friend dynamic_fws_multimap_base;

    using size_type = size_type;
    using value_type = T;
    using iterator = typename mcd::vector<T>::iterator;
    using const_iterator = typename mcd::vector<T>::const_iterator;

    template <bool IsConst = Const, typename = std::enable_if_t<IsConst>>
    explicit bucket(bucket<false> const& b)
        : multimap_{b.multimap_}, index_{b.index_} {}

    size_type index() const { return index_; }
    size_type size() const { return get_index().size_; }
    size_type capacity() const { return get_index().capacity_; }
    [[nodiscard]] bool empty() const { return size() == 0; }

    iterator begin() {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
                 .data_.begin() +
             get_index().begin_;
    }

    const_iterator begin() const {
      return multimap_.data_.begin() + get_index().begin_;
    }

    iterator end() {
      auto const& index = get_index();
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
                 .data_.begin() +
             index.begin_ + index.size_;
    }

    const_iterator end() const {
      auto const& index = get_index();
      return multimap_.data_.begin() + index.begin_ + index.size_;
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    friend iterator begin(bucket& b) { return b.begin(); }
    friend const_iterator begin(bucket const& b) { return b.begin(); }
    friend iterator end(bucket& b) { return b.end(); }
    friend const_iterator end(bucket const& b) { return b.end(); }

    T& operator[](size_type index) {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .data_[get_index().begin_ + index];
    }

    T const& operator[](size_type index) const {
      return multimap_.data_[get_index().begin_ + index];
    }

    T& at(size_type index) {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .data_[get_and_check_data_index(index)];
    }

    T const& at(size_type index) const {
      return multimap_.data_[get_and_check_data_index(index)];
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    size_type push_back(entry_type const& val) {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .push_back_entry(index_, val);
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>,
              typename... Args>
    size_type emplace_back(Args&&... args) {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .emplace_back_entry(index_, std::forward<Args>(args)...);
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    void reserve(size_type new_size) {
      if (new_size > capacity()) {
        const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
            .grow_bucket(index_, get_index(), new_size);
      }
    }

  protected:
    bucket(dynamic_fws_multimap_base const& multimap, size_type index)
        : multimap_(multimap), index_(index) {}

    index_type const& get_index() const { return multimap_.index_[index_]; }

    size_type get_and_check_data_index(size_type index) const {
      auto const& idx = get_index();
      if (index >= idx.size_) {
        throw std::out_of_range{
            "dynamic_fws_multimap::bucket::at() out of range"};
      }
      return idx.begin_ + index;
    }

    dynamic_fws_multimap_base const& multimap_;
    size_type index_;
  };

  using mutable_bucket = bucket<false>;
  using const_bucket = bucket<true>;

  template <bool Const>
  struct bucket_iterator {
    friend dynamic_fws_multimap_base;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = bucket<Const>;
    using difference_type = int;
    using pointer = value_type;
    using reference = value_type;

    template <bool IsConst = Const, typename = std::enable_if_t<IsConst>>
    explicit bucket_iterator(bucket_iterator<false> const& it)
        : multimap_{it.multimap_}, index_{it.index_} {}

    value_type operator*() const { return multimap_.at(index_); }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    value_type operator*() {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .at(index_);
    }

    value_type operator->() const { return multimap_.at(index_); }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    value_type operator->() {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .at(index_);
    }

    bucket_iterator& operator+=(difference_type n) {
      index_ += n;
      return *this;
    }

    bucket_iterator& operator-=(difference_type n) {
      index_ -= n;
      return *this;
    }

    bucket_iterator& operator++() {
      ++index_;
      return *this;
    }

    bucket_iterator& operator--() {
      ++index_;
      return *this;
    }

    bucket_iterator operator+(difference_type n) const {
      return {multimap_, index_ + n};
    }

    bucket_iterator operator-(difference_type n) const {
      return {multimap_, index_ - n};
    }

    int operator-(bucket_iterator const& rhs) const {
      return index_ - rhs.index_;
    };

    value_type operator[](difference_type n) const {
      return multimap_.at(index_ + n);
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    value_type operator[](difference_type n) {
      return const_cast<dynamic_fws_multimap_base&>(multimap_)  // NOLINT
          .at(index_ + n);
    }

    bool operator<(bucket_iterator const& rhs) const {
      return index_ < rhs.index_;
    }
    bool operator<=(bucket_iterator const& rhs) const {
      return index_ <= rhs.index_;
    }
    bool operator>(bucket_iterator const& rhs) const {
      return index_ > rhs.index_;
    }
    bool operator>=(bucket_iterator const& rhs) const {
      return index_ >= rhs.index_;
    }

    bool operator==(bucket_iterator const& rhs) const {
      return &multimap_ == &rhs.multimap_ && index_ == rhs.index_;
    }

    bool operator!=(bucket_iterator const& rhs) const {
      return &multimap_ != &rhs.multimap_ || index_ != rhs.index_;
    }

  protected:
    bucket_iterator(dynamic_fws_multimap_base const& multimap, size_type index)
        : multimap_(multimap), index_(index) {}

    dynamic_fws_multimap_base const& multimap_;
    size_type index_;
  };

  using iterator = bucket_iterator<false>;
  using const_iterator = bucket_iterator<true>;

  mutable_bucket operator[](size_type index) {
    if (index >= index_.size()) {
      index_.resize(index + 1);
    }
    return {*this, index};
  }

  const_bucket operator[](size_type index) const { return {*this, index}; }

  mutable_bucket at(size_type index) {
    if (index >= index_.size()) {
      throw std::out_of_range{"dynamic_fws_multimap::at() out of range"};
    } else {
      return {*this, index};
    }
  }

  const_bucket at(size_type index) const {
    if (index >= index_.size()) {
      throw std::out_of_range{"dynamic_fws_multimap::at() out of range"};
    } else {
      return {*this, index};
    }
  }

  mutable_bucket emplace_back() { return this[index_size()]; }

  size_type index_size() const { return index_.size(); }
  size_type data_size() const { return index_.size(); }
  size_type element_count() const { return element_count_; }
  [[nodiscard]] bool empty() const { return index_size() == 0; }

  iterator begin() { return {*this, 0}; }
  const_iterator begin() const { return {*this, 0}; }
  iterator end() { return iterator{*this, index_.size()}; }
  const_iterator end() const { return const_iterator{*this, index_.size()}; }

  friend iterator begin(dynamic_fws_multimap_base const& m) {
    return m.begin();
  }

  friend iterator end(dynamic_fws_multimap_base const& m) { return m.end(); }

  mcd::vector<T>& data() { return data_; }

protected:
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
        std::max(static_cast<size_type>(idx.capacity_ + 1),
                 idx.capacity_ == 0 ? initial_capacity_
                                    : idx.capacity_ * growth_factor_);
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

  void move_entries(size_type const map_index, size_type const old_data_index,
                    size_type const new_data_index, size_type const count) {
    auto old_data = &data_[old_data_index];
    auto new_data = &data_[new_data_index];
    for (auto i = static_cast<size_type>(0); i < count;
         ++i, ++old_data, ++new_data) {
      new (new_data) T(std::move(*old_data));
      old_data->~T();
    }
    static_cast<Derived&>(*this).entries_moved(map_index, old_data_index,
                                               new_data_index, count);
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
  size_type initial_capacity_{1};
  size_type growth_factor_{2};
};

template <typename T, typename SizeType = std::uint32_t>
struct dynamic_fws_multimap
    : public dynamic_fws_multimap_base<dynamic_fws_multimap<T, SizeType>, T,
                                       SizeType> {
  void entries_moved(SizeType const /*map_index*/,
                     SizeType const /*old_data_index*/,
                     SizeType const /*new_data_index*/,
                     SizeType const /*count*/) {}
};

}  // namespace motis
