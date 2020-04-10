#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

namespace motis::parking::prepare {

struct station {
  station() = default;
  station(std::string id, geo::latlng pos) : id_(std::move(id)), pos_(pos) {}

  std::string id_;
  geo::latlng pos_;
};

struct stations {
  explicit stations(std::string const& schedule_path);

  std::vector<std::pair<station, double>> get_in_radius(
      geo::latlng const& center, double radius) const;

  std::size_t size() const { return stations_.size(); }

private:
  std::vector<station> stations_;
  geo::point_rtree geo_index_;
};

}  // namespace motis::parking::prepare
