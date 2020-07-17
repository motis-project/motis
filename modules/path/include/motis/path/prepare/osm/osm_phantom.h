#pragma once

#include <mutex>
#include <vector>

#include "geo/latlng.h"

#include "motis/path/prepare/osm/osm_way.h"
#include "motis/path/prepare/osm/segment_rtree.h"
#include "motis/path/prepare/schedule/stations.h"

namespace motis::path {

struct osm_node_phantom {
  osm_node_phantom() = default;
  osm_node_phantom(int64_t const id, geo::latlng const pos)
      : id_{id}, pos_{pos} {}

  std::vector<size_t> way_idx_;
  int64_t id_{0};
  geo::latlng pos_;
};

struct osm_edge_phantom {
  osm_edge_phantom(size_t const way_idx, size_t const offset,
                   geo::latlng const from, geo::latlng const to)
      : way_idx_{way_idx}, offset_{offset}, segment_{from, to} {}
  size_t way_idx_;
  size_t offset_;
  std::pair<geo::latlng, geo::latlng> segment_;
};

struct osm_node_phantom_match {
  osm_node_phantom phantom_;

  double distance_{std::numeric_limits<double>::infinity()};
  station const* station_{nullptr};
};

struct osm_edge_phantom_match {
  void locate();

  osm_edge_phantom phantom_;

  double distance_{std::numeric_limits<double>::infinity()};
  station const* station_{nullptr};
  geo::latlng station_pos_;  // maybe station itself or stop position

  geo::latlng pos_{};
  double along_track_dist_{std::numeric_limits<double>::infinity()};
  bool eq_from_{false}, eq_to_{false};
};

struct osm_phantom_builder {
  osm_phantom_builder(station_index const&, mcd::vector<osm_way> const&);

  void build_osm_phantoms(station const*);

  std::pair<std::vector<osm_node_phantom_match>,
            std::vector<osm_edge_phantom_match>>
  match_osm_phantoms(station const*, geo::latlng const&, double radius) const;

  void append_phantoms(std::vector<osm_node_phantom_match> const&,
                       std::vector<osm_edge_phantom_match> const&);

  void finalize();

  station_index const& station_idx_;
  mcd::vector<osm_way> const& osm_ways_;

  std::vector<station const*> matched_stations_;

  std::vector<osm_node_phantom> node_phantoms_;
  geo::point_rtree node_phantom_rtree_;

  std::vector<osm_edge_phantom> edge_phantoms_;
  segment_rtree edge_phantom_rtree_;

  std::mutex mutex_;
  std::vector<osm_node_phantom_match> n_phantoms_;
  std::vector<osm_edge_phantom_match> e_phantoms_;
};

std::pair<std::vector<osm_node_phantom_match>,
          std::vector<osm_edge_phantom_match>>
make_phantoms(station_index const&, mcd::vector<osm_way> const&);

}  // namespace motis::path
