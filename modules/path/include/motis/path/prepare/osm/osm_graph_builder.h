#pragma once

#include <map>
#include <string>

#include "motis/path/prepare/schedule/stations.h"

#include "motis/path/prepare/osm/osm_graph.h"
#include "motis/path/prepare/osm/osm_way.h"

namespace motis::path {

struct osm_graph_builder {
  osm_graph_builder(osm_graph& graph, station_index const& station_idx)
      : graph_{graph}, station_idx_{station_idx} {}

  void build_graph(mcd::vector<mcd::vector<osm_way>> const&);
  void add_component(mcd::vector<osm_way> const&);

  std::mutex mutex_;

  osm_graph& graph_;
  station_index const& station_idx_;
};

}  // namespace motis::path
