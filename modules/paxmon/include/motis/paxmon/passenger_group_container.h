#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>

#include "utl/erase.h"
#include "utl/verify.h"

#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"

#include "motis/paxmon/allocator.h"
#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/group_route.h"
#include "motis/paxmon/index_types.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reroute_log_entry.h"

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
    auto [g_ptr, m_ptr] = allocator_.create(pg);
    groups_.emplace_back(g_ptr);
    m_ptr->id_ = id;
    groups_by_source_[m_ptr->source_].emplace_back(id);

    auto const routes = group_routes_.emplace_back();
    utl::verify(routes.index() == id,
                "passenger_group_container: group_routes out of sync");
    auto const reroute_log_entries = reroute_log_entries_.emplace_back();
    utl::verify(reroute_log_entries.index() == id,
                "passenger_group_container: reroute_log_entries out of sync");

    ++active_groups_;
    return m_ptr;
  }

  inline void release(passenger_group_index const pgi) {
    auto const ptr = groups_.at(pgi);
    if (ptr) {
      auto const* m_ptr = allocator_.get(ptr);
      utl::erase(groups_by_source_[m_ptr->source_], pgi);

      for (auto const& gr : group_routes_.at(pgi)) {
        route_edges_.at(gr.edges_index_).clear();
        compact_journeys_.at(gr.compact_journey_index_).clear();
      }
      group_routes_.at(pgi).clear();

      for (auto const& rle : reroute_log_entries_.at(pgi)) {
        log_entry_new_routes_.at(rle.index_).clear();
      }
      reroute_log_entries_.at(pgi).clear();

      allocator_.release(ptr);
      groups_[pgi] = {};
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

  passenger_group& group(passenger_group_index const index) {
    return *at(index);
  }

  passenger_group const& group(passenger_group_index const index) const {
    return *at(index);
  }

  fws_compact_journey journey(compact_journey_index const cji) const {
    return fws_compact_journey{compact_journeys_.at(cji)};
  }

  auto routes(passenger_group_index const pgi) const {
    return group_routes_.at(pgi);
  }

  auto routes(passenger_group_index const pgi) { return group_routes_.at(pgi); }

  auto reroute_log_entries(passenger_group_index const pgi) const {
    return reroute_log_entries_.at(pgi);
  }

  auto reroute_log_entries(passenger_group_index const pgi) {
    return reroute_log_entries_.at(pgi);
  }

  group_route const& route(passenger_group_with_route const pgwr) const {
    return routes(pgwr.pg_).at(pgwr.route_);
  }

  group_route& route(passenger_group_with_route const pgwr) {
    return routes(pgwr.pg_).at(pgwr.route_);
  }

  auto route_edges(group_route_edges_index const edges_index) const {
    return route_edges_.at(edges_index);
  }

  auto route_edges(group_route_edges_index const edges_index) {
    return route_edges_.at(edges_index);
  }

  // TODO(groups): remove custom allocator
  allocator<passenger_group> allocator_;
  std::vector<group_pointer> groups_;
  mcd::hash_map<data_source, mcd::vector<passenger_group_index>>
      groups_by_source_;
  std::size_t active_groups_{};

  // index: passenger_group_index
  dynamic_fws_multimap<group_route> group_routes_;

  // index: compact_journey_index
  dynamic_fws_multimap<journey_leg> compact_journeys_;

  // index: group_route_edges_index
  dynamic_fws_multimap<edge_index> route_edges_;

  // index: passenger_group_index
  dynamic_fws_multimap<reroute_log_entry> reroute_log_entries_;
  // index: reroute_log_entry_index
  dynamic_fws_multimap<reroute_log_route_info> log_entry_new_routes_;
};

}  // namespace motis::paxmon
