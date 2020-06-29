#include "motis/path/polyline_builder.h"

#include "tiles/fixed/convert.h"

namespace motis::path {

void double_polyline_builder::append(
    bool const is_fwd, path_database_query::resolvable_feature const* rf) {
  is_extra_ = is_extra_ || rf->is_extra_;

  auto const& l = mpark::get<tiles::fixed_polyline>(rf->geometry_);
  utl::verify(rf->feature_id_ == std::numeric_limits<uint64_t>::max() ||
                  (l.size() == 1 && l.front().size() > 1),
              "double_polyline_builder: invalid line geometry");

  auto const append = [&](auto const& p) {
    auto const ll = tiles::fixed_to_latlng(p);
    if (coords_.empty() || coords_[coords_.size() - 2] != ll.lat_ ||
        coords_[coords_.size() - 1] != ll.lng_) {
      coords_.push_back(ll.lat_);
      coords_.push_back(ll.lng_);
    }
  };

  if (is_fwd) {
    std::for_each(std::begin(l.front()), std::end(l.front()), append);
  } else {
    std::for_each(std::rbegin(l.front()), std::rend(l.front()), append);
  }
}

[[nodiscard]] bool double_polyline_builder::empty() const {
  return coords_.empty();
}

void double_polyline_builder::clear() {
  is_extra_ = false;
  coords_.clear();
}

void double_polyline_builder::finish() {
  if (coords_.size() == 2) {  // was a fallback segment
    coords_.push_back(coords_[0]);
    coords_.push_back(coords_[1]);
  }

  utl::verify(coords_.size() >= 4,
              "double_polyline_builder: invalid polyline size: {}",
              coords_.size());
}

void google_polyline_builder::append(
    bool const is_fwd, path_database_query::resolvable_feature const* rf) {
  is_extra_ = is_extra_ || rf->is_extra_;

  auto const& l = mpark::get<tiles::fixed_polyline>(rf->geometry_);
  utl::verify(rf->feature_id_ == std::numeric_limits<uint64_t>::max() ||
                  (l.size() == 1 && l.front().size() > 1),
              "google_polyline_builder: invalid line geometry");

  auto const append = [&](auto const& p) {
    if (enc_.push_nonzero_diff(tiles::fixed_to_latlng(p))) {
      ++count_;
    }
  };

  if (is_fwd) {
    std::for_each(std::begin(l.front()), std::end(l.front()), append);
  } else {
    std::for_each(std::rbegin(l.front()), std::rend(l.front()), append);
  }
}

[[nodiscard]] bool google_polyline_builder::empty() const {
  return count_ == 0;
}

void google_polyline_builder::clear() {
  is_extra_ = false;
  count_ = 0;
  enc_.reset();
}

void google_polyline_builder::finish() {
  if (count_ == 1) {  // was a fallback segment -> repeat last
    enc_.push_difference(0);
    enc_.push_difference(0);
    ++count_;
  }

  utl::verify(count_ >= 2, "google_polyline_builder: invalid polyline {}",
              count_);
}

}  // namespace motis::path