#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "utl/erase.h"

#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/paxmon/allocator.h"
#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct passenger_group_container {
  using group_pointer = typename allocator<passenger_group>::pointer;

  template <bool Const>
  struct group_iterator {
    friend passenger_group_container;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = passenger_group;
    using pointer = std::conditional_t<Const, const value_type*, value_type*>;
    using reference = std::conditional_t<Const, const value_type&, value_type&>;

    template <bool IsConst = Const, typename = std::enable_if_t<IsConst>>
    explicit group_iterator(group_iterator<false> const& it)
        : pgc_{it.pgc_}, index_{it.index_} {}

    pointer operator*() const { return pgc_[index_]; }
    pointer operator->() const { return pgc_[index_]; }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    pointer operator*() {
      return const_cast<passenger_group_container&>(pgc_)[index_];  // NOLINT
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    pointer operator->() {
      return const_cast<passenger_group_container&>(pgc_)[index_];  // NOLINT
    }

    group_iterator& operator++() {
      ++index_;
      return *this;
    }

    group_iterator operator++(int) {
      auto old = *this;
      ++(*this);
      return old;
    }

    group_iterator& operator--() {
      --index_;
      return *this;
    }

    group_iterator operator--(int) {
      auto old = *this;
      --(*this);
      return old;
    }

    group_iterator& operator+=(difference_type n) {
      index_ = static_cast<passenger_group_index>(index_ + n);
      return *this;
    }

    group_iterator& operator-=(difference_type n) {
      index_ = static_cast<passenger_group_index>(index_ - n);
      return *this;
    }

    group_iterator operator+(difference_type n) const {
      return {pgc_, static_cast<passenger_group_index>(index_ + n)};
    }

    group_iterator operator-(difference_type n) const {
      return {pgc_, static_cast<passenger_group_index>(index_ - n)};
    }

    difference_type operator-(group_iterator const& rhs) const {
      return static_cast<difference_type>(index_) -
             static_cast<difference_type>(rhs.index_);
    }

    pointer operator[](difference_type n) const { return pgc_[index_ + n]; }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    pointer operator[](difference_type n) {
      return const_cast<passenger_group_container&>(  // NOLINT
          pgc_)[index_ + n];
    }

    friend bool operator==(group_iterator const& lhs,
                           group_iterator const& rhs) {
      return lhs.index_ == rhs.index_ && &lhs.pgc_ == &rhs.pgc_;
    }

    friend bool operator!=(group_iterator const& lhs,
                           group_iterator const& rhs) {
      return lhs.index_ != rhs.index_ || &lhs.pgc_ != &rhs.pgc_;
    }

    friend bool operator<(group_iterator const& lhs,
                          group_iterator const& rhs) {
      return lhs.index_ < rhs.index_;
    }

    friend bool operator<=(group_iterator const& lhs,
                           group_iterator const& rhs) {
      return lhs.index_ <= rhs.index_;
    }

    friend bool operator>(group_iterator const& lhs,
                          group_iterator const& rhs) {
      return lhs.index_ > rhs.index_;
    }

    friend bool operator>=(group_iterator const& lhs,
                           group_iterator const& rhs) {
      return lhs.index_ >= rhs.index_;
    }

  protected:
    group_iterator(passenger_group_container const& pgc,
                   passenger_group_index index)
        : pgc_{pgc}, index_{index} {}

    passenger_group_container const& pgc_;
    passenger_group_index index_{};
  };

  using iterator = group_iterator<false>;
  using const_iterator = group_iterator<true>;

  inline passenger_group* add(passenger_group&& pg) {
    auto const id = static_cast<passenger_group_index>(groups_.size());
    auto [g_ptr, m_ptr] = allocator_.create(std::move(pg));
    groups_.emplace_back(g_ptr);
    m_ptr->id_ = id;
    groups_by_source_[m_ptr->source_].emplace_back(id);
    ++active_groups_;
    return m_ptr;
  }

  inline void release(passenger_group_index const id) {
    auto const ptr = groups_.at(id);
    if (ptr) {
      auto const* m_ptr = allocator_.get(ptr);
      utl::erase(groups_by_source_[m_ptr->source_], id);
      allocator_.release(ptr);
      groups_[id] = {};
      --active_groups_;
    }
  }

  passenger_group* operator[](passenger_group_index const index) {
    auto const ptr = groups_[index];
    return ptr ? allocator_.get(ptr) : nullptr;
  }

  passenger_group const* operator[](passenger_group_index const index) const {
    auto const ptr = groups_[index];
    return ptr ? allocator_.get(ptr) : nullptr;
  }

  passenger_group* at(passenger_group_index const index) {
    return allocator_.get_checked(groups_.at(index));
  }

  passenger_group const* at(passenger_group_index const index) const {
    return allocator_.get_checked(groups_.at(index));
  }

  iterator begin() { return {*this, 0}; }
  const_iterator begin() const { return {*this, 0}; }
  friend iterator begin(passenger_group_container& pgc) { return pgc.begin(); }
  friend const_iterator begin(passenger_group_container const& pgc) {
    return pgc.begin();
  }

  iterator end() {
    return {*this, static_cast<passenger_group_index>(groups_.size())};
  }
  const_iterator end() const {
    return {*this, static_cast<passenger_group_index>(groups_.size())};
  }
  friend iterator end(passenger_group_container& pgc) { return pgc.end(); }
  friend const_iterator end(passenger_group_container const& pgc) {
    return pgc.end();
  }

  std::size_t size() const { return groups_.size(); }
  std::size_t active_groups() const { return active_groups_; }

  void reserve(std::size_t size) { groups_.reserve(size); }

  allocator<passenger_group> allocator_;
  std::vector<group_pointer> groups_;
  mcd::hash_map<data_source, mcd::vector<passenger_group_index>>
      groups_by_source_;
  std::size_t active_groups_{};
};

}  // namespace motis::paxmon
