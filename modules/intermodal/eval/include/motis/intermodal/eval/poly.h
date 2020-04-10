#pragma once

#include <limits>
#include <random>

#include "boost/geometry/algorithms/covered_by.hpp"
#include "boost/geometry/algorithms/envelope.hpp"
#include "boost/geometry/geometries/geometries.hpp"

#include "motis/intermodal/eval/bounds.h"

namespace motis::intermodal::eval {

class poly : public bounds {
public:
  using polygon_t = boost::geometry::model::polygon<geo::latlng>;
  using multi_polygon_t = boost::geometry::model::multi_polygon<polygon_t>;
  using box_t = boost::geometry::model::box<geo::latlng>;

  explicit poly(const multi_polygon_t& poly) : poly_(poly) {
    box_ = boost::geometry::return_envelope<box_t>(poly);
    std::random_device rd;
    mt_.seed(rd());
    lon_dist_ = std::uniform_real_distribution<double>(box_.min_corner().lng_,
                                                       box_.max_corner().lng_);
    lat_dist_ = std::uniform_real_distribution<double>(box_.min_corner().lat_,
                                                       box_.max_corner().lat_);
  }

  ~poly() override = default;

  poly(poly const&) = delete;
  poly& operator=(poly const&) = delete;
  poly(poly&&) = delete;
  poly& operator=(poly&&) = delete;

  bool contains(geo::latlng const& loc) const override {
    return boost::geometry::covered_by(loc, poly_);
  }

  geo::latlng random_pt() override {
    geo::latlng pt;
    do {
      pt = {lat_dist_(mt_), lon_dist_(mt_)};
    } while (!boost::geometry::covered_by(pt, poly_));
    return pt;
  }

private:
  multi_polygon_t poly_;
  box_t box_;
  std::mt19937 mt_;
  std::uniform_real_distribution<double> lon_dist_;
  std::uniform_real_distribution<double> lat_dist_;
};

}  // namespace motis::intermodal::eval
