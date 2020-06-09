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

  trip_iterator(trip const* t, int const i) : trip_(t), index_(i) {}

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
  trip const* trip_;
  int index_;
};

struct sections {
  using iterator = trip_iterator<trip_section>;

  explicit sections(trip const* t) : t_(t) {}

  iterator begin() const { return begin(t_); }
  iterator end() const { return end(t_); }

  friend iterator begin(sections const& s) { return begin(s.t_); }
  friend iterator end(sections const& s) { return end(s.t_); }

  static iterator begin(trip const* t) { return {t, 0}; }
  static iterator end(trip const* t) {
    return {t, static_cast<int>(t->edges_->size())};
  }

  trip const* t_;
};

struct stops {
  using iterator = trip_iterator<trip_stop>;

  explicit stops(trip const* t) : t_(t) {}

  iterator begin() const { return begin(t_); }
  iterator end() const { return end(t_); }

  friend iterator begin(stops const& s) { return begin(s.t_); }
  friend iterator end(stops const& s) { return end(s.t_); }

  static iterator begin(trip const* t) { return {t, 0}; }
  static iterator end(trip const* t) {
    return {t, static_cast<int>(t->edges_->size()) + 1};
  }

  trip const* t_;
};

}  // namespace motis::access
