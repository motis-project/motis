#pragma once

#include <limits>
#include <random>

#include "boost/geometry/algorithms/covered_by.hpp"
#include "boost/geometry/geometries/box.hpp"

#include "motis/intermodal/eval/bounds.h"

namespace motis::intermodal::eval {

class bbox : public bounds {
public:
  bbox()
      : bbox(boost::geometry::model::box<geo::latlng>(
            geo::latlng{std::numeric_limits<double>::min(),
                        std::numeric_limits<double>::min()},
            geo::latlng{std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max()})) {}

  explicit bbox(boost::geometry::model::box<geo::latlng> box) : box_(box) {
    std::random_device rd;
    mt_.seed(rd());
    lon_dist_ = std::uniform_real_distribution<double>(box_.min_corner().lng_,
                                                       box_.max_corner().lng_);
    lat_dist_ = std::uniform_real_distribution<double>(box_.min_corner().lat_,
                                                       box_.max_corner().lat_);
  }

  ~bbox() override = default;

  bbox(bbox const&) = delete;
  bbox& operator=(bbox const&) = delete;
  bbox(bbox&&) = delete;
  bbox& operator=(bbox&&) = delete;

  bool contains(geo::latlng const& loc) const override {
    return boost::geometry::covered_by(loc, box_);
  }

  geo::latlng random_pt() override { return {lat_dist_(mt_), lon_dist_(mt_)}; }

private:
  boost::geometry::model::box<geo::latlng> box_;
  std::mt19937 mt_;
  std::uniform_real_distribution<double> lon_dist_;
  std::uniform_real_distribution<double> lat_dist_;
};

}  // namespace motis::intermodal::eval
