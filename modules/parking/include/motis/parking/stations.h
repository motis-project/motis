#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "motis/core/schedule/schedule.h"

namespace motis::parking {

struct station_info {
  station_info() = default;
  station_info(std::string id, geo::latlng pos)
      : id_(std::move(id)), pos_(pos) {}

  std::string id_;
  geo::latlng pos_;
};

struct stations {
  explicit stations(schedule const& sched);
  explicit stations(std::vector<station_info> const& stations);

  std::vector<std::pair<station_info, double>> get_in_radius(
      geo::latlng const& center, double radius) const;

  std::size_t size() const { return stations_.size(); }

  std::vector<station_info> get_stations() { return stations_; }

private:
  std::vector<station_info> stations_;
  geo::point_rtree geo_index_;
};

}  // namespace motis::parking
