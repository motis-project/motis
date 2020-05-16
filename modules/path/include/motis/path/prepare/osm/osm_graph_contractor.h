#pragma once

#include <queue>
#include <vector>

#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

#include "utl/erase_if.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm/osm_graph.h"

namespace motis::path {

struct osm_graph_dist {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(osm_graph_dist)
  size_t from_, to_, dist_{0UL};
};

struct osm_graph_contractor {
  struct contract_edge {
    contract_edge(size_t node_idx, size_t dist)
        : node_idx_{node_idx}, dist_{dist} {}

    size_t node_idx_, dist_;  // to for out_edges_, from for inc_edges_
  };

  struct contract_node {
    bool is_terminal_ = false;
    std::vector<contract_edge> out_edges_, inc_edges_;
  };

  explicit osm_graph_contractor(osm_graph const& graph) : graph_{graph} {
    mutex_ = std::vector<std::mutex>(graph_.nodes_.size());
    nodes_ = utl::to_vec(graph_.nodes_, [](auto const&) {
      return std::make_unique<contract_node>();
    });

    for (auto const& n : graph_.nodes_) {
      auto from = n->idx_;
      for (auto const& e : n->edges_) {
        auto to = e.to_->idx_;
        nodes_[from]->out_edges_.emplace_back(to, e.dist_);
        nodes_[to]->inc_edges_.emplace_back(from, e.dist_);
      }
    }

    for (auto& node : nodes_) {  // todo maybe parallel?
      auto less = [](auto const& a, auto const& b) {
        return std::tie(a.node_idx_, a.dist_) < std::tie(b.node_idx_, b.dist_);
      };
      auto eq = [](auto const& a, auto const& b) {
        return a.node_idx_ == b.node_idx_;
      };

      utl::erase_duplicates(node->out_edges_, less, eq);
      utl::erase_duplicates(node->inc_edges_, less, eq);
    }

    for (auto const& link :
         graph_.node_station_links_) {  // todo maybe parallel?
      nodes_[link.node_idx_]->is_terminal_ = true;
    }
  }

  size_t ops_ = 0;
  void contract() {
    std::queue<size_t> queue;
    for (auto i = 0UL; i < nodes_.size(); ++i) {
      if (nodes_[i]->is_terminal_) {
        queue.push(i);
      }
    }

    while (!queue.empty()) {
      if (++ops_ % 100000 == 0) {
        std::clog << ops_ << std::endl;
      }

      auto const node_idx = queue.front();
      queue.pop();

      // lock node_idx
      if (!nodes_[node_idx]) {
        continue;
      }

      auto& node = *nodes_[node_idx];
      auto neighbors = get_neighbors(node);
      // lock neighbors with timeout

      for (auto const& neigbor_idx : neighbors) {
        auto const& cn = nodes_.at(neigbor_idx);
        if (!cn || cn->is_terminal_ ||
            cn->inc_edges_.size() + cn->out_edges_.size() >= 100) {
          continue;
        }
        queue.push(neigbor_idx);
      }

      if (neighbors.size() >= 100) {
        continue;
      }

      process_node(node);

      if (!node.is_terminal_) {
        nodes_[node_idx] = std::unique_ptr<contract_node>();
      }
    }
  }

  static std::vector<size_t> get_neighbors(contract_node const& curr) {
    std::vector<size_t> neighbors;
    for (auto const& e : curr.inc_edges_) {
      neighbors.push_back(e.node_idx_);
    }
    for (auto const& e : curr.out_edges_) {
      neighbors.push_back(e.node_idx_);
    }
    utl::erase_duplicates(neighbors);
    return neighbors;
  }

  void process_node(contract_node& curr) {
    utl::erase_if(curr.inc_edges_,
                  [&](auto const& e) { return !nodes_[e.node_idx_]; });
    utl::erase_if(curr.out_edges_,
                  [&](auto const& e) { return !nodes_[e.node_idx_]; });

    auto const maintain_shortcut = [&](auto& edges, auto node_idx, auto dist) {
      auto it = std::find_if(begin(edges), end(edges), [&](auto const& e) {
        return e.node_idx_ == node_idx;
      });

      if (it == end(edges)) {
        edges.emplace_back(node_idx, dist);
      } else {
        it->dist_ = std::min(it->dist_, dist);
      }
    };

    for (auto const& inc : curr.inc_edges_) {
      if (!nodes_[inc.node_idx_]) {
        continue;  // already dropped
      }

      for (auto const& out : curr.out_edges_) {
        if (!nodes_[out.node_idx_] || inc.node_idx_ == out.node_idx_) {
          continue;  // already dropped or same node
        }

        auto d = inc.dist_ + out.dist_;
        maintain_shortcut(nodes_[inc.node_idx_]->out_edges_, out.node_idx_, d);
        maintain_shortcut(nodes_[out.node_idx_]->inc_edges_, inc.node_idx_, d);
      }
    }
  }

  std::vector<osm_graph_dist> collect_distances() {
    std::vector<osm_graph_dist> distances;
    for (auto i = 0UL; i < nodes_.size(); ++i) {
      auto const& cn = nodes_.at(i);
      if (!cn) {
        continue;
      }
      for (auto const& e : cn->out_edges_) {
        if (!nodes_[e.node_idx_]) {
          continue;
        }
        // verify(nodes_[e.node_idx_]->is_terminal_, "non-terminal remains");
        distances.emplace_back(osm_graph_dist{i, e.node_idx_, e.dist_});
      }
    }
    std::sort(begin(distances), end(distances));
    return distances;
  }

  osm_graph const& graph_;

  // parallel vectors
  std::vector<std::mutex> mutex_;
  std::vector<std::unique_ptr<contract_node>> nodes_;
};

inline osm_graph contract_graph(osm_graph const& orig) {
  motis::logging::scoped_timer t{"contract_graph"};
  osm_graph contr;
  contr.components_ = orig.components_;
  contr.nodes_.resize(orig.nodes_.size());
  contr.component_offsets_ = orig.component_offsets_;

  auto const get_or_create_node = [&](auto const idx) {
    auto& from = contr.nodes_[idx];
    if (!from) {
      auto const& o_from = *orig.nodes_[idx];
      from = std::make_unique<osm_node>(o_from.idx_, o_from.component_id_,
                                        o_from.osm_id_, o_from.pos_);
    }
    return from.get();
  };

  for (auto const& link : orig.node_station_links_) {
    get_or_create_node(link.node_idx_);
  }

  osm_graph_contractor contractor{orig};
  contractor.contract();
  for (auto const& graph_dist : contractor.collect_distances()) {
    auto* from = get_or_create_node(graph_dist.from_);
    auto* to = get_or_create_node(graph_dist.to_);
    from->edges_.emplace_back(0, true, graph_dist.dist_, from, to);
  }

  return contr;
}

}  // namespace motis::path
