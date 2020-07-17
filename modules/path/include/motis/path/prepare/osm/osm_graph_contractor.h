#pragma once

#include <atomic>
#include <vector>

#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

#include "motis/path/prepare/osm/osm_graph.h"

namespace motis::path {

struct osm_graph_dist {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(osm_graph_dist)
  size_t from_{0UL}, to_{0UL}, dist_{0UL};
};

using contract_cluster_id_t = size_t;
using contract_node_idx_t = size_t;
using contract_distance_t = size_t;

constexpr auto const kNoClusterId =
    std::numeric_limits<contract_cluster_id_t>::max();

constexpr auto const kInvalidClusterId =
    std::numeric_limits<contract_cluster_id_t>::max() - 1;

struct osm_graph_contractor {
  struct contract_edge {
    contract_edge(size_t node_idx, size_t dist)
        : node_idx_{node_idx}, dist_{dist} {}

    size_t node_idx_, dist_;  // to for out_edges_, from for inc_edges_
  };

  struct contract_node {
    bool is_terminal_{false};
    std::vector<contract_edge> out_edges_, inc_edges_;

    std::atomic<contract_cluster_id_t> cluster_id_{kNoClusterId};
  };

  struct contract_task {
    size_t offset_, node_count_, task_id_, task_count_;
  };

  explicit osm_graph_contractor(osm_graph const&);

  void contract();
  void contract(contract_task const&);

  static std::vector<size_t> get_neighbors(contract_node const&);

  void process_node(contract_node&);
  std::vector<osm_graph_dist> collect_distances();

  osm_graph const& graph_;
  size_t ops_{0};

  std::vector<std::unique_ptr<contract_node>> nodes_;
};

osm_graph contract_graph(osm_graph const&);

}  // namespace motis::path
