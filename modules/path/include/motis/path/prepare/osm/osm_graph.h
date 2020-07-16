#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "geo/latlng.h"
#include "geo/polyline.h"
#include "geo/xyz.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

struct osm_node;
struct osm_edge;

struct node_station_link {
  node_station_link(std::string station_id, size_t node_idx, double distance)
      : station_id_{std::move(station_id)},
        node_idx_{node_idx},
        distance_{distance} {}

  std::string station_id_;
  size_t node_idx_;
  double distance_;
};

struct osm_graph {
  size_t components_{0ULL};
  std::vector<std::unique_ptr<osm_node>> nodes_;
  std::vector<osm_path> paths_;

  std::vector<node_station_link> node_station_links_;

  // component_id -> (offset, node_count)
  std::vector<std::pair<size_t, size_t>> component_offsets_;
};

struct osm_node {
  osm_node(size_t idx, size_t component_id, int64_t osm_id, geo::latlng pos)
      : idx_{idx},
        component_id_{component_id},
        osm_id_{osm_id},
        pos_{pos},
        xyz_{pos} {}

  size_t idx_;
  size_t component_id_;

  int64_t osm_id_;
  geo::latlng pos_;
  geo::xyz xyz_;

  std::vector<osm_edge> edges_;
};

struct osm_edge {
  osm_edge(uint64_t polyline_idx, bool forward, size_t dist,
           osm_node const* from, osm_node const* to)
      : polyline_idx_{polyline_idx},
        forward_{static_cast<size_t>(forward)},
        dist_{dist},
        from_{from},
        to_{to} {}

  bool is_forward() const { return forward_ != 0U; }

  uint64_t polyline_idx_ : 63;
  size_t forward_ : 1;

  size_t dist_;  // xyz distance!

  osm_node const* from_;
  osm_node const* to_;
};

inline void print_osm_graph_stats(source_spec const& source_spec,
                                  osm_graph const& graph) {
  namespace ml = motis::logging;
  LOG(ml::info) << "osm graph stats " << source_spec.str();

  auto const node_count = std::count_if(begin(graph.nodes_), end(graph.nodes_),
                                        [](auto const& n) { return !!n; });
  if (node_count == 0) {
    LOG(ml::info) << "- GRAPH EMPTY!";
    return;
  }
  LOG(ml::info) << "- nodes: " << node_count;

  std::vector<size_t> vec;
  for (auto const& n : graph.nodes_) {
    if (n) {
      vec.push_back(n->edges_.size());
    }
  }

  auto const count = std::accumulate(begin(vec), end(vec), size_t{});
  auto const avg = count / vec.size();

  LOG(ml::info) << "- edges: " << count;
  LOG(ml::info) << "- degree: "  //
                << " 0:" << std::count(begin(vec), end(vec), 0)
                << " 1:" << std::count(begin(vec), end(vec), 1)
                << " 2:" << std::count(begin(vec), end(vec), 2)
                << " 3:" << std::count(begin(vec), end(vec), 3)
                << " 4:" << std::count(begin(vec), end(vec), 4);

  std::sort(begin(vec), end(vec));
  LOG(ml::info) << "- degree: "  //
                << " avg:" << avg  //
                << " q75:" << vec[0.75 * (vec.size() - 1)]  //
                << " q90:" << vec[0.90 * (vec.size() - 1)]  //
                << " q95:" << vec[0.95 * (vec.size() - 1)];

  LOG(ml::info) << "- node_station_links: " << graph.node_station_links_.size();
}

}  // namespace motis::path
