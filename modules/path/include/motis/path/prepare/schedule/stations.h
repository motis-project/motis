#pragma once

#include <string>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "motis/path/prepare/osm/osm_data.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

struct station {
  station() = default;

  station(std::string id, std::string name, geo::latlng pos)
      : id_{std::move(id)}, name_{std::move(name)}, pos_{pos} {}

  friend bool operator<(station const& lhs, station const& rhs) {
    return lhs.id_ < rhs.id_;
  }

  friend bool operator==(station const& lhs, station const& rhs) {
    return lhs.id_ == rhs.id_;
  }

  std::string id_;

  std::string name_;
  std::set<service_class> classes_;

  geo::latlng pos_;
  std::vector<geo::latlng> stop_positions_;
};

struct station_index {
  std::vector<station> stations_;
  geo::point_rtree index_;
};

station_index collect_stations(mcd::vector<station_seq> const&);
station_index make_station_index(std::vector<station>);

void annotate_stop_positions(osm_data const&, station_index&);

}  // namespace motis::path
