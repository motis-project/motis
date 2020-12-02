#pragma once

#include <cstdint>
#include <iterator>
#include <tuple>

#include "utl/verify.h"

#include "motis/core/schedule/trip.h"
#include "motis/core/access/trip_section.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/graph.h"

namespace motis::paxmon {

struct trip_section_with_load {
  trip_section_with_load(trip const* trp, trip_data const* td, int const idx)
      : section_{trp, idx},
        edge_{td == nullptr ? nullptr : td->edges_.at(idx)} {}

  inline bool has_load_info() const { return edge_ != nullptr; }

  inline bool has_capacity_info() const {
    return edge_ != nullptr && edge_->has_capacity();
  }

  inline std::uint16_t capacity() const {
    return edge_ != nullptr ? edge_->capacity() : 0;
  }

  std::uint16_t base_load() const {
    return edge_ != nullptr ? get_base_load(edge_->pax_connection_info_) : 0;
  }

  std::uint16_t mean_load() const {
    return edge_ != nullptr ? get_mean_load(edge_->pax_connection_info_) : 0;
  }

  pax_pdf load_pdf() const {
    return edge_ != nullptr ? get_load_pdf(edge_->pax_connection_info_)
                            : pax_pdf{};
  }

  pax_cdf load_cdf() const { return get_cdf(load_pdf()); }

  std::uint16_t median_load() const {
    return edge_ != nullptr ? get_median_load(load_cdf()) : 0;
  }

  motis::access::trip_section section_;
  edge const* edge_{};
};

struct trip_section_load_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int;
  using value_type = trip_section_with_load;
  using pointer = value_type;
  using reference = value_type;

  trip_section_load_iterator(trip const* trp, trip_data const* td,
                             int const idx)
      : trip_{trp}, td_{td}, index_{idx} {}

  trip_section_with_load operator*() { return {trip_, td_, index_}; }
  trip_section_with_load operator->() { return {trip_, td_, index_}; }
  trip_section_with_load operator[](int rhs) {
    return {trip_, td_, index_ + rhs};
  }

  trip_section_load_iterator& operator+=(int rhs) {
    index_ += rhs;
    return *this;
  }

  trip_section_load_iterator& operator-=(int rhs) {
    index_ -= rhs;
    return *this;
  }

  trip_section_load_iterator& operator++() {
    ++index_;
    return *this;
  }

  trip_section_load_iterator& operator--() {
    --index_;
    return *this;
  }

  trip_section_load_iterator& operator++(int) {
    trip_section_load_iterator tmp{*this};
    ++index_;
    return tmp;
  }

  trip_section_load_iterator& operator--(int) {
    trip_section_load_iterator tmp{*this};
    --index_;
    return tmp;
  }

  trip_section_load_iterator operator+(difference_type rhs) const {
    return {trip_, td_, index_ + rhs};
  }

  trip_section_load_iterator operator-(difference_type rhs) const {
    return {trip_, td_, index_ + rhs};
  }

  friend trip_section_load_iterator operator+(
      difference_type lhs, trip_section_load_iterator const& rhs) {
    return {rhs.trip_, rhs.td_, rhs.index_ + lhs};
  }

  friend trip_section_load_iterator operator-(
      difference_type lhs, trip_section_load_iterator const& rhs) {
    return {rhs.trip_, rhs.td_, rhs.index_ - lhs};
  }

  difference_type operator-(trip_section_load_iterator const& rhs) const {
    return index_ - rhs.index_;
  }

  bool operator==(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) ==
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

  bool operator!=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) !=
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

  bool operator<(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) <
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

  bool operator<=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) <=
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

  bool operator>(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) >
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

  bool operator>=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, td_, index_) >=
           std::tie(rhs.trip_, rhs.td_, rhs.index_);
  }

protected:
  trip const* trip_{};
  trip_data const* td_{};
  int index_{};
};

struct sections_with_load {
  using iterator = trip_section_load_iterator;

  sections_with_load(graph const& g, trip const* trp) : trip_{trp} {
    if (auto const it = g.trip_data_.find(trp); it != std::end(g.trip_data_)) {
      td_ = it->second.get();
      utl::verify(trip_->edges_->size() == td_->edges_.size(),
                  "motis trip edge count ({}) != paxmon trip edge count ({})",
                  trip_->edges_->size(), td_->edges_.size());
    }
  }

  inline bool has_load_info() const { return td_ != nullptr; }

  iterator begin() const { return {trip_, td_, 0}; }
  iterator end() const {
    return {trip_, td_, static_cast<int>(trip_->edges_->size())};
  }

  friend iterator begin(sections_with_load const& s) { return s.begin(); }
  friend iterator end(sections_with_load const& s) { return s.end(); }

  trip const* trip_{};
  trip_data const* td_{};
};

}  // namespace motis::paxmon
