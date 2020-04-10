#pragma once

#include <limits>
#include <optional>
#include <ostream>
#include <vector>

#include "geo/polyline.h"

#include "utl/concat.h"
#include "utl/repeat_n.h"
#include "utl/verify.h"

namespace motis::path {

constexpr int64_t kPathUnknownNodeId = -1;

struct osm_path {
  osm_path() = default;

  explicit osm_path(size_t const size) {
    polyline_.reserve(size);
    osm_node_ids_.reserve(size);
  }

  osm_path(geo::polyline polyline, std::vector<int64_t> osm_node_ids)
      : polyline_(std::move(polyline)), osm_node_ids_(std::move(osm_node_ids)) {
    verify_path();
  }

  explicit osm_path(geo::polyline polyline)
      : polyline_(std::move(polyline)),
        osm_node_ids_(utl::repeat_n(kPathUnknownNodeId, polyline_.size())) {}

  void append(osm_path const& other) {
    utl::concat(polyline_, other.polyline_);
    utl::concat(osm_node_ids_, other.osm_node_ids_);
    verify_path();
  }

  void reverse() {
    std::reverse(begin(polyline_), end(polyline_));
    std::reverse(begin(osm_node_ids_), end(osm_node_ids_));
  }

  osm_path partial(size_t begin, size_t end) const {
    utl::verify(begin < polyline_.size() && end > 0 &&
                    end <= polyline_.size() && begin < end,
                "osm_path: partial invalid arguments");

    osm_path result;
    result.polyline_ = {std::begin(polyline_) + begin,
                        std::begin(polyline_) + end};
    result.osm_node_ids_ = {std::begin(osm_node_ids_) + begin,
                            std::begin(osm_node_ids_) + end};
    return result;
  }

  osm_path partial_replaced_padded(
      size_t begin, size_t end,  //
      std::optional<geo::latlng> const& left_replace,
      std::optional<geo::latlng> const& right_pad) const {
    utl::verify(begin < polyline_.size() && end > 0 &&
                    end <= polyline_.size() && begin < end,
                "osm_path: partial invalid arguments");

    osm_path result(end - begin + (right_pad ? 1U : 0U));

    result.polyline_ = {std::begin(polyline_) + begin,
                        std::begin(polyline_) + end};
    result.osm_node_ids_ = {std::begin(osm_node_ids_) + begin,
                            std::begin(osm_node_ids_) + end};

    if (left_replace) {
      result.polyline_.front() = *left_replace;
      result.osm_node_ids_.front() = kPathUnknownNodeId;
    }

    if (right_pad) {
      result.polyline_.push_back(*right_pad);
      result.osm_node_ids_.push_back(kPathUnknownNodeId);
    }

    return result;
  }

  void unique();
  void remove_loops();

  size_t size() const { return polyline_.size(); }

  void verify_path() const {
    utl::verify(polyline_.size() == osm_node_ids_.size(),
                "osm_path inconsistent");
  }

  friend bool operator==(osm_path const& a, osm_path const& b) {
    return std::tie(a.polyline_, a.osm_node_ids_) ==
           std::tie(b.polyline_, b.osm_node_ids_);
  }

  friend std::ostream& operator<<(std::ostream& os, osm_path const& p) {
    auto old_precision = os.precision(11);
    for (auto i = 0UL; i < p.size(); ++i) {
      os << "[" << p.polyline_.at(i).lng_ << ", " << p.polyline_.at(i).lat_
         << "] " << p.osm_node_ids_.at(i) << "\n";
    }
    os.precision(old_precision);
    return os;
  }

  geo::polyline polyline_;
  std::vector<int64_t> osm_node_ids_;
};

}  // namespace motis::path
