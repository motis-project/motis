#pragma once

#include "motis/vector.h"

#include <cassert>
#include <iterator>
#include <type_traits>

namespace motis {

template <typename T, typename Index = uint64_t>
struct fws_multimap_entry {
  static_assert(!std::is_same_v<std::remove_cv<T>, bool>, "bool not supported");
  using iterator = typename mcd::vector<T>::const_iterator;

  fws_multimap_entry(mcd::vector<T> const& data,
                     mcd::vector<Index> const& index, Index key)
      : data_(data),
        index_start(index[key]),
        index_end(index[key + 1]),
        key_{key} {}

  fws_multimap_entry(mcd::vector<T> const& data, Index start_index,
                     Index end_index)
      : data_(data), index_start(start_index), index_end(end_index) {}

  inline iterator begin() const { return data_.begin() + index_start; }
  inline iterator end() const { return data_.begin() + index_end; }

  inline iterator cbegin() const { return begin(); }
  inline iterator cend() const { return end(); }

  friend iterator begin(fws_multimap_entry<T, Index> const& e) {
    return e.begin();
  }
  friend iterator end(fws_multimap_entry<T, Index> const& e) { return e.end(); }

  inline T const& operator[](std::size_t index) const {
    return data_[data_index(index)];
  }

  inline std::size_t data_index(std::size_t index) const {
    assert(index_start + index < data_.size());
    return index_start + index;
  }

  inline std::size_t size() const { return index_end - index_start; }
  inline bool empty() const { return size() == 0; }

  inline Index key() const { return key_; }

  mcd::vector<T> const& data_;
  Index const index_start;
  Index const index_end;
  Index key_;
};

template <typename MapType, typename EntryType>
struct fws_multimap_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = EntryType;
  using difference_type = int;
  using pointer = value_type*;
  using reference = value_type&;

  fws_multimap_iterator(MapType const& map, std::size_t index)
      : map_(map), index_(index) {}

  value_type operator*() const { return {map_[index_]}; };

  fws_multimap_iterator<MapType, EntryType>& operator+=(int n) {
    index_ += n;
    return *this;
  }

  fws_multimap_iterator<MapType, EntryType>& operator-=(int n) {
    index_ -= n;
    return *this;
  }

  fws_multimap_iterator<MapType, EntryType>& operator++() {
    ++index_;
    return *this;
  }

  fws_multimap_iterator<MapType, EntryType>& operator--() {
    --index_;
    return *this;
  }

  fws_multimap_iterator<MapType, EntryType> operator+(int n) const {
    return {map_, index_ + n};
  }

  fws_multimap_iterator<MapType, EntryType> operator-(int n) const {
    return {map_, index_ + n};
  }

  int operator-(const fws_multimap_iterator<MapType, EntryType>& rhs) const {
    return index_ - rhs.index_;
  }

  value_type& operator[](int n) const { return {map_, index_ + n}; }

  bool operator<(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return index_ < rhs.index_;
  }

  bool operator>(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return index_ > rhs.index_;
  }

  bool operator<=(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return index_ <= rhs.index_;
  }

  bool operator>=(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return index_ >= rhs.index_;
  }

  bool operator==(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return &map_ == &rhs.map_ && index_ == rhs.index_;
  }

  bool operator!=(fws_multimap_iterator<MapType, EntryType> const& rhs) const {
    return !(*this == rhs);
  }

protected:
  MapType const& map_;
  std::size_t index_;
};

template <typename T, typename Index = uint64_t>
struct fws_multimap {
  using iterator = fws_multimap_iterator<fws_multimap<T, Index>,
                                         fws_multimap_entry<T, Index>>;

  inline void push_back(T const& val) {
    assert(!complete_);
    data_.push_back(val);
  }

  template <typename... Args>
  inline void emplace_back(Args&&... args) {
    data_.emplace_back(std::forward<Args>(args)...);
  }

  inline Index current_key() const { return static_cast<Index>(index_.size()); }

  inline void finish_key() {
    assert(!complete_);
    index_.push_back(current_start_);
    current_start_ = static_cast<Index>(data_.size());
  }

  inline void skip_to_key(Index dest) {
    assert(!complete_);
    assert(current_key() <= dest);
    while (current_key() < dest) {
      finish_key();
    }
  }

  inline void finish_map() {
    assert(!complete_);
    index_.push_back(static_cast<Index>(data_.size()));
    complete_ = true;
  }

  inline void reserve_index(Index size) {
    index_.reserve(static_cast<std::size_t>(size) + 1);
  }

  inline fws_multimap_entry<T, Index> operator[](Index index) const {
    assert(index < index_.size() - 1);
    return fws_multimap_entry<T, Index>{data_, index_, index};
  }

  inline iterator begin() const { return {*this, 0}; }
  inline iterator end() const { return {*this, index_.size() - 1}; }

  inline iterator cbegin() const { return begin(); }
  inline iterator cend() const { return end(); }

  friend iterator begin(fws_multimap<T, Index> const& e) { return e.begin(); }
  friend iterator end(fws_multimap<T, Index> const& e) { return e.end(); }

  inline std::size_t index_size() const { return index_.size(); }
  inline std::size_t data_size() const { return data_.size(); }
  inline bool finished() const { return complete_; }

  mcd::vector<T> data_;
  mcd::vector<Index> index_;
  Index current_start_{0};
  bool complete_{false};
};

template <typename T, typename Index = uint64_t>
struct shared_idx_fws_multimap {
  using iterator = fws_multimap_iterator<shared_idx_fws_multimap<T, Index>,
                                         fws_multimap_entry<T, Index>>;

  explicit shared_idx_fws_multimap(mcd::vector<Index> const& base_index)
      : base_index_(base_index) {}

  inline void push_back(T const& val) { data_.push_back(val); }

  template <typename... Args>
  inline void emplace_back(Args&&... args) {
    data_.emplace_back(std::forward<Args>(args)...);
  }

  inline Index current_key() const {
    return static_cast<Index>(base_index_.size());
  }

  inline void finish_key() {}

  inline void skip_to_key(Index /*dest*/) {}

  inline void finish_map() {}

  inline void reserve_index(std::size_t) {}

  inline fws_multimap_entry<T, Index> operator[](Index index) const {
    assert(index < base_index_.size() - 1);
    return {data_, base_index_, index};
  }

  inline iterator begin() const { return {*this, 0}; }
  inline iterator end() const { return {*this, base_index_.size() - 1}; }

  inline iterator cbegin() const { return begin(); }
  inline iterator cend() const { return end(); }

  friend iterator begin(shared_idx_fws_multimap const& e) { return e.begin(); }

  friend iterator end(shared_idx_fws_multimap const& e) { return e.end(); }

  inline std::size_t index_size() const { return base_index_.size(); }
  inline std::size_t data_size() const { return data_.size(); }

  mcd::vector<Index> const& base_index_;
  mcd::vector<T> data_;
};

template <typename T, typename Index = uint64_t>
struct nested_fws_multimap {
  explicit nested_fws_multimap(mcd::vector<Index> const& base_index)
      : base_index_(base_index) {}

  inline void finish_base_key() {}

  inline void push_back(T const& val) {
    assert(!complete_);
    data_.push_back(val);
  }

  template <typename... Args>
  inline void emplace_back(Args&&... args) {
    data_.emplace_back(std::forward<Args>(args)...);
  }

  inline Index current_key() const { return static_cast<Index>(index_.size()); }

  inline void finish_nested_key() {
    assert(!complete_);
    index_.push_back(current_start_);
    current_start_ = data_.size();
  }

  inline void finish_map() {
    assert(!complete_);
    index_.push_back(static_cast<Index>(data_.size()));
    complete_ = true;
  }

  inline void reserve_index(std::size_t size) { index_.reserve(size + 1); }
  inline void reserve_data(std::size_t size) { data_.reserve(size); }

  inline fws_multimap_entry<T, Index> at(Index outer_index,
                                         Index inner_index) const {
    assert(static_cast<std::size_t>(outer_index) < base_index_.size() - 1);
    auto const start_idx = index_[base_index_[outer_index] + inner_index];
    auto const end_idx = index_[base_index_[outer_index] + inner_index + 1];
    return {data_, start_idx, end_idx};
  }

  std::size_t index_size() const { return index_.size(); }
  std::size_t data_size() const { return data_.size(); }
  bool finished() const { return complete_; }

  mcd::vector<Index> const& base_index_;
  mcd::vector<Index> index_;
  mcd::vector<T> data_;
  Index current_start_{0};
  bool complete_{false};
};

}  // namespace motis
