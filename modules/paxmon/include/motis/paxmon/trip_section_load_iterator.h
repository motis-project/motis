#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <tuple>

#include "utl/verify.h"

#include "motis/core/schedule/trip.h"
#include "motis/core/access/trip_section.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/get_load.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

enum class capacity_info_source : std::uint8_t {
  TRIP_OR_LOOKUP,  // current trip, lookup if trip is not tracked by paxmon
  TRIP,  // current trip capacity, unknown if trip is not tracked by paxmon
  LOOKUP  // ignore current trip, always lookup capacity
};

struct trip_section_with_load {
  trip_section_with_load(schedule const& sched, universe const& uv,
                         trip const* trp, trip_data_index const tdi,
                         capacity_info_source const cs, int const idx)
      : sched_{sched},
        uv_{uv},
        section_{trp, idx},
        edge_{tdi == INVALID_TRIP_DATA_INDEX
                  ? nullptr
                  : uv.trip_data_.edges(tdi).at(idx).get(uv)} {

    auto const get_trip_capacity = [this]() {
      if (edge_ != nullptr) {
        capacity_ = edge_->capacity();
        capacity_source_ = edge_->get_capacity_source();
        return true;
      } else {
        return false;
      }
    };

    auto const lookup_capacity = [this, &sched, &uv]() {
      auto const cap = lookup_section_capacity();
      capacity_ = cap.capacity_.seats();
      capacity_source_ = cap.source_;
    };

    switch (cs) {
      case capacity_info_source::TRIP_OR_LOOKUP:
        if (!get_trip_capacity()) {
          lookup_capacity();
        }
        break;
      case capacity_info_source::TRIP: get_trip_capacity(); break;
      case capacity_info_source::LOOKUP: lookup_capacity(); break;
    }
  }

  inline bool has_paxmon_data() const { return edge_ != nullptr; }
  inline bool has_load_info() const { return has_paxmon_data(); }

  inline bool has_capacity_info() const { return capacity_ != 0; }

  inline std::uint16_t capacity() const { return capacity_; }

  inline capacity_source get_capacity_source() const {
    return capacity_source_;
  }

  inline section_capacity lookup_section_capacity(bool detailed = false) const {
    return get_capacity(sched_, section_.lcon(), section_.ev_key_from(),
                        section_.ev_key_to(), uv_.capacity_maps_, detailed);
  }

  std::uint16_t base_load() const {
    return edge_ != nullptr
               ? get_base_load(
                     uv_.passenger_groups_,
                     uv_.pax_connection_info_.group_routes(edge_->pci_))
               : 0;
  }

  std::uint16_t mean_load() const {
    return edge_ != nullptr
               ? get_mean_load(
                     uv_.passenger_groups_,
                     uv_.pax_connection_info_.group_routes(edge_->pci_))
               : 0;
  }

  pax_pdf load_pdf() const {
    return edge_ != nullptr
               ? get_load_pdf(
                     uv_.passenger_groups_,
                     uv_.pax_connection_info_.group_routes(edge_->pci_))
               : pax_pdf{};
  }

  pax_cdf load_cdf() const { return get_cdf(load_pdf()); }

  std::uint16_t median_load() const {
    return edge_ != nullptr ? get_median_load(load_cdf()) : 0;
  }

  light_connection const& lcon() const { return section_.lcon(); }

  ev_key ev_key_from() const { return section_.ev_key_from(); }
  ev_key ev_key_to() const { return section_.ev_key_to(); }

  edge const* paxmon_edge() const { return edge_; }

  schedule const& sched_;
  universe const& uv_;
  motis::access::trip_section section_;
  edge const* edge_{};
  std::uint16_t capacity_{};
  capacity_source capacity_source_{capacity_source::UNKNOWN};
};

struct trip_section_load_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int;
  using value_type = trip_section_with_load;
  using pointer = value_type;
  using reference = value_type;

  trip_section_load_iterator(schedule const& sched, universe const& uv,
                             trip const* trp, trip_data_index const tdi,
                             capacity_info_source const cs, int const idx)
      : sched_{sched}, uv_{uv}, trip_{trp}, tdi_{tdi}, cs_{cs}, index_{idx} {}

  trip_section_with_load operator*() {
    return {sched_, uv_, trip_, tdi_, cs_, index_};
  }
  trip_section_with_load operator->() {
    return {sched_, uv_, trip_, tdi_, cs_, index_};
  }
  trip_section_with_load operator[](int rhs) {
    return {sched_, uv_, trip_, tdi_, cs_, index_ + rhs};
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

  trip_section_load_iterator operator++(int) {
    trip_section_load_iterator tmp{*this};
    ++index_;
    return tmp;
  }

  trip_section_load_iterator operator--(int) {
    trip_section_load_iterator tmp{*this};
    --index_;
    return tmp;
  }

  trip_section_load_iterator operator+(difference_type rhs) const {
    return {sched_, uv_, trip_, tdi_, cs_, index_ + rhs};
  }

  trip_section_load_iterator operator-(difference_type rhs) const {
    return {sched_, uv_, trip_, tdi_, cs_, index_ + rhs};
  }

  friend trip_section_load_iterator operator+(
      difference_type lhs, trip_section_load_iterator const& rhs) {
    return {rhs.sched_, rhs.uv_, rhs.trip_,
            rhs.tdi_,   rhs.cs_, rhs.index_ + lhs};
  }

  friend trip_section_load_iterator operator-(
      difference_type lhs, trip_section_load_iterator const& rhs) {
    return {rhs.sched_, rhs.uv_, rhs.trip_,
            rhs.tdi_,   rhs.cs_, rhs.index_ - lhs};
  }

  difference_type operator-(trip_section_load_iterator const& rhs) const {
    return index_ - rhs.index_;
  }

  bool operator==(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) ==
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

  bool operator!=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) !=
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

  bool operator<(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) <
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

  bool operator<=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) <=
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

  bool operator>(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) >
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

  bool operator>=(trip_section_load_iterator const& rhs) const {
    return std::tie(trip_, tdi_, index_) >=
           std::tie(rhs.trip_, rhs.tdi_, rhs.index_);
  }

protected:
  schedule const& sched_;
  universe const& uv_;
  trip const* trip_{};
  trip_data_index tdi_{};
  capacity_info_source cs_{};
  int index_{};
};

struct sections_with_load {
  using iterator = trip_section_load_iterator;

  sections_with_load(
      schedule const& sched, universe const& uv, trip const* trp,
      trip_data_index const tdi,
      capacity_info_source const cs = capacity_info_source::TRIP_OR_LOOKUP)
      : sched_{sched}, uv_{uv}, trip_{trp}, tdi_{tdi}, cs_{cs} {
    if (tdi_ != INVALID_TRIP_DATA_INDEX) {
      auto const td_edges = uv.trip_data_.edges(tdi_);
      utl::verify(trip_->edges_->size() == td_edges.size(),
                  "motis trip edge count ({}) != paxmon trip edge count ({})",
                  trip_->edges_->size(), td_edges.size());
    }
  }

  sections_with_load(
      schedule const& sched, universe const& uv, trip const* trp,
      capacity_info_source const cs = capacity_info_source::TRIP_OR_LOOKUP)
      : sections_with_load{sched, uv, trp,
                           uv.trip_data_.find_index(trp->trip_idx_), cs} {}

  inline std::size_t size() const { return trip_->edges_->size(); }
  inline bool empty() const { return trip_->edges_->empty(); }

  inline bool has_paxmon_data() const {
    return tdi_ != INVALID_TRIP_DATA_INDEX;
  }

  inline bool has_load_info() const { return has_paxmon_data(); }

  iterator begin() const { return {sched_, uv_, trip_, tdi_, cs_, 0}; }
  iterator end() const {
    return {sched_, uv_, trip_,
            tdi_,   cs_, static_cast<int>(trip_->edges_->size())};
  }

  friend iterator begin(sections_with_load const& s) { return s.begin(); }
  friend iterator end(sections_with_load const& s) { return s.end(); }

  trip_section_with_load operator[](std::size_t idx) const {
    return begin()[static_cast<int>(idx)];
  }

  trip_section_with_load at(std::size_t idx) const {
    if (idx < size()) {
      return (*this)[idx];
    } else {
      throw std::out_of_range{"trip_section_with_load::at(): out of range"};
    }
  }

  trip_section_with_load front() const { return at(0); }
  trip_section_with_load back() const { return at(size() - 1); }

  schedule const& sched_;
  universe const& uv_;
  trip const* trip_{};
  trip_data_index tdi_{};
  capacity_info_source cs_{};
};

}  // namespace motis::paxmon
