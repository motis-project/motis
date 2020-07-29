#include "motis/path/prepare/osm/osm_graph_contractor.h"

#include <deque>
#include <queue>
#include <thread>

#include "boost/range/irange.hpp"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/parallel_for.h"

#include "motis/hash_map.h"

#include "motis/core/common/flat_matrix.h"
#include "motis/core/common/floyd_warshall.h"
#include "motis/core/common/logging.h"

namespace motis::path {

osm_graph_contractor::osm_graph_contractor(osm_graph const& graph)
    : graph_{graph} {
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

  utl::parallel_for(boost::irange(nodes_.size()), [&](auto const& i) {
    auto const less = [](auto const& a, auto const& b) {
      return std::tie(a.node_idx_, a.dist_) < std::tie(b.node_idx_, b.dist_);
    };
    auto const eq = [](auto const& a, auto const& b) {
      return a.node_idx_ == b.node_idx_;
    };

    utl::erase_duplicates(nodes_[i]->out_edges_, less, eq);
    utl::erase_duplicates(nodes_[i]->inc_edges_, less, eq);
  });

  for (auto const& link : graph_.node_station_links_) {
    nodes_[link.node_idx_]->is_terminal_ = true;
  }
}

void osm_graph_contractor::contract() {
  auto const concurrency =
      static_cast<size_t>(std::thread::hardware_concurrency());

  std::vector<contract_task> work;
  work.reserve(std::max(graph_.component_offsets_.size(), concurrency));
  for (auto const& [offset, node_count] : graph_.component_offsets_) {
    auto const component_task_count =
        std::min(std::max<size_t>(node_count / 10000, 1), concurrency);
    for (auto i = 0ULL; i < component_task_count; ++i) {
      work.push_back({offset, node_count, i, component_task_count});
    }
  }

  utl::parallel_for(work, [&](auto const& w) { contract(w); });
}

struct contract_cluster {
  contract_cluster(osm_graph_contractor& contractor, size_t cluster_id)
      : contractor_{contractor}, cluster_id_{cluster_id} {}

  void find_nodes(contract_node_idx_t const origin_idx) {
    auto const& origin = contractor_.nodes_[origin_idx];
    contract_cluster_id_t origin_cluster_id = kNoClusterId;
    auto const origin_added = origin->cluster_id_.compare_exchange_strong(
        origin_cluster_id, cluster_id_);
    if (!origin_added) {
      return;  // already part of another cluster
    }

    std::deque<contract_node_idx_t> queue;
    queue.emplace_back(origin_idx);
    while (!queue.empty() && outer_node_count_ + queue.size() < 100) {
      auto curr_idx = queue.front();
      queue.pop_front();

      auto const& curr = contractor_.nodes_[curr_idx];

      auto any_blocked = false;
      for_each_edge(*curr, [&](auto const& e) {
        auto& other = contractor_.nodes_[e.node_idx_];
        utl::verify(other != nullptr, "found a deleted node: {}", e.node_idx_);

        contract_cluster_id_t other_cluster_id = kNoClusterId;
        auto const other_added = other->cluster_id_.compare_exchange_strong(
            other_cluster_id, cluster_id_);

        // not added now and not added previously
        any_blocked =
            any_blocked || (!other_added && (other_cluster_id != cluster_id_));

        if (other_added) {
          queue.emplace_back(e.node_idx_);
        }
      });

      auto const is_inner_node = !curr->is_terminal_ && !any_blocked;
      outer_node_count_ += is_inner_node ? 0 : 1;
      cluster_nodes_.emplace_back(is_inner_node, curr_idx);
    }

    // check if this is actually an inner node
    for (auto const node_idx : queue) {
      ++outer_node_count_;
      cluster_nodes_.emplace_back(false, node_idx);
    }

    // first: is_inner_node == false
    std::sort(begin(cluster_nodes_), end(cluster_nodes_));
  }

  template <typename Fn>
  void for_each_edge(osm_graph_contractor::contract_node const& n, Fn&& fn) {
    for (auto const& e : n.out_edges_) {
      fn(e);
    }
    for (auto const& e : n.inc_edges_) {
      fn(e);
    }
  }

  void compute_distances() {
    mcd::hash_map<contract_node_idx_t, size_t> node_idx_to_mat_idx;
    for (auto const& [i, pair] : utl::enumerate(cluster_nodes_)) {
      node_idx_to_mat_idx[pair.second] = i;
    }

    constexpr auto const kInvalidDistance =
        std::numeric_limits<uint32_t>::max();
    auto mat =
        make_flat_matrix<uint32_t>(cluster_nodes_.size(), kInvalidDistance);

    for (auto const& [i, pair] : utl::enumerate(cluster_nodes_)) {
      mat(i, i) = 0;

      auto const& node = contractor_.nodes_[pair.second];
      for (auto const& e : node->out_edges_) {
        auto const it = node_idx_to_mat_idx.find(e.node_idx_);

        // either not is_inner_node or other must be part of this cluster
        utl::verify(!pair.first || it != end(node_idx_to_mat_idx),
                    "contract_cluster::compute_distances missing node ");

        if (it != end(node_idx_to_mat_idx)) {
          mat(i, it->second) = e.dist_;
        }
      }
    }

    floyd_warshall(mat);

    // mark inner nodes as deleted
    for (auto i = outer_node_count_; i < cluster_nodes_.size(); ++i) {
      contractor_.nodes_.at(cluster_nodes_.at(i).second)->cluster_id_ =
          kInvalidClusterId;
    }

    // update outer nodes
    for (auto i = 0ULL; i < outer_node_count_; ++i) {
      auto& curr_node = contractor_.nodes_.at(cluster_nodes_.at(i).second);
      auto& out_edges = curr_node->out_edges_;
      auto& inc_edges = curr_node->inc_edges_;

      // remove edges to removed (inner) nodes
      utl::erase_if(out_edges, [&](auto const& e) {
        return contractor_.nodes_.at(e.node_idx_)->cluster_id_ ==
               kInvalidClusterId;
      });
      utl::erase_if(inc_edges, [&](auto const& e) {
        return contractor_.nodes_.at(e.node_idx_)->cluster_id_ ==
               kInvalidClusterId;
      });

      auto const old_out_edge_count = out_edges.size();
      auto const old_inc_edge_count = inc_edges.size();

      for (auto j = 0ULL; j < outer_node_count_; ++j) {
        if (i == j) {
          continue;
        }
        if (auto const out_dist = mat(i, j); out_dist != kInvalidDistance) {
          out_edges.emplace_back(cluster_nodes_.at(j).second, out_dist);
        }
        if (auto const inc_dist = mat(j, i); inc_dist != kInvalidDistance) {
          inc_edges.emplace_back(cluster_nodes_.at(j).second, inc_dist);
        }
      }

      auto const finalize = [&](auto& edges, auto const old_edge_count) {
        std::inplace_merge(begin(edges), begin(edges) + old_edge_count,
                           end(edges), [](auto const& lhs, auto const& rhs) {
                             return lhs.node_idx_ < rhs.node_idx_;
                           });
        utl::equal_ranges_linear(
            edges,
            [](auto const& lhs, auto const& rhs) {
              return lhs.node_idx_ == rhs.node_idx_;
            },
            [&](auto lb, auto ub) {
              if (std::distance(lb, ub) == 1) {
                return;
              }

              auto const min_it = std::min_element(
                  lb, ub, [](auto const& lhs, auto const& rhs) {
                    return lhs.dist_ < rhs.dist_;
                  });
              for (auto it = lb; it != ub; ++it) {
                if (it != min_it) {
                  it->dist_ = kInvalidDistance;
                }
              }
            });
        utl::erase_if(
            edges, [&](auto const& e) { return e.dist_ == kInvalidDistance; });
      };

      finalize(out_edges, old_out_edge_count);
      finalize(inc_edges, old_inc_edge_count);
    }
  }

  osm_graph_contractor& contractor_;
  contract_cluster_id_t cluster_id_;
  size_t outer_node_count_{0};

  // (is_inner_node, idx)
  std::vector<std::pair<bool, contract_node_idx_t>> cluster_nodes_;
};

void osm_graph_contractor::contract(contract_task const& task) {
  auto const shift = static_cast<size_t>(task.node_count_ *
                                         static_cast<double>(task.task_id_) /
                                         static_cast<double>(task.task_count_));
  for (auto i = task.offset_; i < task.offset_ + task.node_count_; ++i) {
    auto const j = (i + shift) % task.node_count_;
    if (!nodes_[j]->is_terminal_ || nodes_[j]->cluster_id_ != kNoClusterId) {
      continue;
    }

    contract_cluster cc{*this, j};
    cc.find_nodes(j);
    cc.compute_distances();
  }
}

std::vector<osm_graph_dist> osm_graph_contractor::collect_distances() {
  std::vector<osm_graph_dist> distances;
  for (auto i = 0UL; i < nodes_.size(); ++i) {
    auto const& cn = nodes_.at(i);
    if (cn->cluster_id_ == kInvalidClusterId) {
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

osm_graph contract_graph(osm_graph const& orig) {
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
