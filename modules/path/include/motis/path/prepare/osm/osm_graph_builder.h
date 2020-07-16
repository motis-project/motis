#pragma once

#include <map>
#include <string>

#include "motis/path/prepare/schedule/stations.h"

#include "motis/path/prepare/osm/osm_graph.h"
#include "motis/path/prepare/osm/osm_phantom.h"
#include "motis/path/prepare/osm/osm_way.h"

namespace motis::path {

struct osm_graph_builder {
  osm_graph_builder(osm_graph& graph, source_spec const source_spec,
                    station_index const& station_idx)
      : graph_{graph}, source_spec_{source_spec}, station_idx_{station_idx} {}

  void build_graph(mcd::vector<mcd::vector<osm_way>> const&);

  void add_component(mcd::vector<osm_way> const&);
  void add_component(mcd::vector<osm_way> const&,
                     std::vector<osm_node_phantom_match> const&,
                     std::vector<osm_edge_phantom_match> const&);

  double get_penalty_factor(source_bits) const;

  std::mutex mutex_;

  osm_graph& graph_;
  source_spec source_spec_;
  station_index const& station_idx_;
};

}  // namespace motis::path
