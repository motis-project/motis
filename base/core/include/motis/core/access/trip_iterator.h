#pragma once

#include <iterator>
#include <tuple>

#include "motis/core/schedule/trip.h"
#include "motis/core/access/trip_section.h"
#include "motis/core/access/trip_stop.h"

namespace motis::access {

template <typename T>
struct trip_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;  // crap
  using difference_type = int;
  using pointer = T*;
  using reference = T&;

  trip_iterator(concrete_trip const t, int const i) : trip_(t), index_(i) {}

  trip_iterator<T>& operator+=(int rhs) {
    index_ += rhs;
    return *this;
  }
  trip_iterator<T>& operator-=(int rhs) {
    index_ -= rhs;
    return *this;
  }

  T operator*() const { return {trip_, index_}; }
  T operator[](int rhs) const { return {trip_, rhs}; }

  trip_iterator<T>& operator++() {
    ++index_;
    return *this;
  }
  trip_iterator<T>& operator--() {
    --index_;
    return *this;
  }
  trip_iterator<T> operator++(int) {
    trip_iterator<T> tmp(*this);
    ++index_;
    return tmp;
  }
  trip_iterator<T> operator--(int) {
    trip_iterator<T> tmp(*this);
    --index_;
    return tmp;
  }
  int operator-(const trip_iterator<T>& rhs) const {
    return index_ - rhs.index_;
  }
  trip_iterator<T> operator+(int rhs) const {
    return trip_iterator<T>(trip_, index_ + rhs);
  }
  trip_iterator<T> operator-(int rhs) const {
    return trip_iterator<T>(trip_, index_ - rhs);
  }
  friend trip_iterator<T> operator+(int lhs, trip_iterator<T> const& rhs) {
    return trip_iterator<T>(rhs.trip_, lhs + rhs.index_);
  }
  friend trip_iterator<T> operator-(int lhs, trip_iterator<T> const& rhs) {
    return trip_iterator<T>(rhs.trip_, lhs - rhs.index_);
  }

  bool operator==(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) == std::tie(rhs.trip_, rhs.index_);
  }
  bool operator!=(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) != std::tie(rhs.trip_, rhs.index_);
  }
  bool operator>(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) > std::tie(rhs.trip_, rhs.index_);
  }
  bool operator<(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) < std::tie(rhs.trip_, rhs.index_);
  }
  bool operator>=(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) >= std::tie(rhs.trip_, rhs.index_);
  }
  bool operator<=(trip_iterator<T> const& rhs) const {
    return std::tie(trip_, index_) <= std::tie(rhs.trip_, rhs.index_);
  }

protected:
  concrete_trip trip_;
  int index_;
};

struct sections {
  using iterator = trip_iterator<trip_section>;

  explicit sections(concrete_trip const t) : t_{t} {}

  iterator begin() const { return begin(t_); }
  iterator end() const { return end(t_); }

  friend iterator begin(sections const& s) { return begin(s.t_); }
  friend iterator end(sections const& s) { return end(s.t_); }

  static iterator begin(concrete_trip t) { return {t, 0}; }
  static iterator end(concrete_trip t) {
    return {t, static_cast<int>(t.trp_->edges_->size())};
  }

  concrete_trip t_;
};

struct stops {
  using iterator = trip_iterator<trip_stop>;

  explicit stops(concrete_trip const t) : t_{t} {}

  iterator begin() const { return begin(t_); }
  iterator end() const { return end(t_); }

  friend iterator begin(stops const& s) { return begin(s.t_); }
  friend iterator end(stops const& s) { return end(s.t_); }

  static iterator begin(concrete_trip const t) { return {t, 0}; }
  static iterator end(concrete_trip const t) {
    auto const& edges = *t.trp_->edges_;
    return {t, edges.empty() ? 0 : static_cast<int>(edges.size() + 1)};
  }

  concrete_trip t_;
};

}  // namespace motis::access
