#pragma once

#include <vector>

#include "geo/latlng.h"

#include "motis/path/prepare/osm/osm_way.h"
#include "motis/path/prepare/schedule/stations.h"

namespace motis::path {

struct osm_node_phantom {
  osm_node_phantom() = default;
  osm_node_phantom(int64_t const id, geo::latlng const pos)
      : id_(id), pos_(pos) {}

  std::vector<size_t> way_idx_;
  int64_t id_ = 0;
  geo::latlng pos_;
};

struct osm_edge_phantom {
  osm_edge_phantom(size_t const way_idx, size_t const offset,
                   geo::latlng const from, geo::latlng const to)
      : way_idx_(way_idx), offset_(offset), segment_(from, to) {}
  size_t way_idx_;
  size_t offset_;
  std::pair<geo::latlng, geo::latlng> segment_;
};

struct osm_node_phantom_with_dist {
  osm_node_phantom_with_dist(osm_node_phantom phantom, double const distance)
      : phantom_(std::move(phantom)), distance_(distance) {}
  osm_node_phantom phantom_;
  double distance_;
};

struct osm_edge_phantom_with_dist {
  osm_edge_phantom_with_dist(osm_edge_phantom phantom, double const distance)
      : phantom_(std::move(phantom)), distance_(distance) {}
  osm_edge_phantom phantom_;
  double distance_;
};

struct located_osm_edge_phantom_with_dist : public osm_edge_phantom_with_dist {
  located_osm_edge_phantom_with_dist(osm_edge_phantom_with_dist phantom,
                                     station const* station,
                                     geo::latlng station_pos)
      : osm_edge_phantom_with_dist{std::move(phantom)},
        station_{station},
        station_pos_{station_pos},
        along_track_dist_{std::numeric_limits<double>::infinity()},
        eq_from_{false},
        eq_to_{false} {}

  void locate();

  station const* station_;
  geo::latlng station_pos_;  // maybe station itself or stop position

  geo::latlng pos_;
  double along_track_dist_;
  bool eq_from_, eq_to_;
};

std::pair<std::vector<std::pair<osm_node_phantom_with_dist, station const*>>,
          std::vector<located_osm_edge_phantom_with_dist>>
make_phantoms(station_index const& station_idx,
              std::vector<size_t> const& matched_stations,
              mcd::vector<osm_way> const& osm_ways);

}  // namespace motis::path
