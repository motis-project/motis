#include "motis/path/prepare/osm/osm_router.h"

#include <queue>

#include "boost/thread/tss.hpp"

#include "utl/erase.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

namespace motis::path {

template <class T, class Container = std::vector<T>,
          class Compare = std::less<typename Container::value_type>>
struct clearable_priority_queue
    : public std::priority_queue<T, Container, Compare> {
  void clear() { this->c.clear(); }
};

struct osm_graph_dijkstra_label {
  osm_graph_dijkstra_label(size_t const idx, size_t const dist,
                           size_t const heuristic_dist, osm_edge const* edge)
      : idx_{idx}, dist_{dist}, heuristic_dist_{heuristic_dist}, edge_{edge} {}

  friend bool operator>(osm_graph_dijkstra_label const& a,
                        osm_graph_dijkstra_label const& b) {
    return a.heuristic_dist_ > b.heuristic_dist_;
  }

  size_t idx_, dist_, heuristic_dist_;
  osm_edge const* edge_;
};

using pq_t = clearable_priority_queue<osm_graph_dijkstra_label,
                                      std::vector<osm_graph_dijkstra_label>,
                                      std::greater<>>;

static boost::thread_specific_ptr<std::vector<size_t>> tls_dists;
static boost::thread_specific_ptr<std::vector<osm_edge const*>> tls_edges;
static boost::thread_specific_ptr<pq_t> tls_pq;

struct osm_graph_dijkstra {
  osm_graph_dijkstra(osm_graph const& graph, size_t initial,
                     std::vector<size_t> const& goals, bool ignore_limit)
      : graph_(graph) {
    if (ignore_limit) {
      limit_ = std::numeric_limits<size_t>::max();
    } else {
      for (auto const& g : goals) {
        auto const dist = geo::distance(graph_.nodes_[initial]->pos_,  //
                                        graph_.nodes_[g]->pos_);
        if (dist > limit_) {
          limit_ = dist;
        }
      }
      limit_ *= 10;
    }

    auto const component_id = graph_.nodes_[initial]->component_id_;
    node_idx_offset_ = graph_.component_offsets_.at(component_id).first;

    auto component_size = graph_.component_offsets_.at(component_id).second;

    if (tls_dists.get() == nullptr) {
      tls_dists.reset(new std::vector<size_t>());
    }
    dists_ = tls_dists.get();
    dists_->clear();
    dists_->resize(component_size, std::numeric_limits<size_t>::max());

    if (tls_edges.get() == nullptr) {
      tls_edges.reset(new std::vector<osm_edge const*>());
    }
    edges_ = tls_edges.get();
    edges_->clear();
    edges_->resize(component_size, nullptr);

    if (tls_pq.get() == nullptr) {
      tls_pq.reset(new pq_t());
    }
    pq_ = tls_pq.get();
    pq_->clear();

    for (auto const& g : goals) {
      if (component_id != graph_.nodes_[g]->component_id_) {
        continue;
      }
      open_goals_.push_back(g);
    }

    if (!open_goals_.empty()) {
      // initialize heuristic goal
      double lat_acc = 0.;
      double lng_acc = 0.;
      for (auto const& g : open_goals_) {
        lat_acc += graph_.nodes_[g]->pos_.lat_;
        lng_acc += graph_.nodes_[g]->pos_.lng_;
      }

      goal_center_ = geo::latlng{lat_acc / open_goals_.size(),
                                 lng_acc / open_goals_.size()};
      for (auto const& g : open_goals_) {
        goal_radius_ = std::max(
            goal_radius_, static_cast<size_t>(std::ceil(geo::haversine_distance(
                              goal_center_, graph_.nodes_[g]->xyz_))));
      }

      // initial label
      dist(initial) = 0;
      pq_->push(osm_graph_dijkstra_label{
          initial, 0, heuristic_distance(initial, 0), nullptr});
    }
  }

  void run() {
    while (!pq_->empty()) {
      auto label = pq_->top();
      pq_->pop();

      auto const this_idx = label.idx_;

      utl::erase(open_goals_, this_idx);
      if (open_goals_.empty()) {
        break;
      }

      auto const& node = graph_.nodes_[this_idx];
      for (auto const& curr_edge : node->edges_) {
        size_t const new_dist = label.dist_ + curr_edge.dist_;
        size_t const to_idx = curr_edge.to_->idx_;

        if (new_dist < limit_ && new_dist < dist(to_idx)) {
          dist(to_idx) = new_dist;
          edge(to_idx) = &curr_edge;
          pq_->push({to_idx, new_dist, heuristic_distance(to_idx, new_dist),
                     &curr_edge});
        }
      }
    }
  }

  std::vector<osm_edge const*> get_edges(size_t const goal) const {
    std::vector<osm_edge const*> result;
    auto curr_edge = edge(goal);
    while (curr_edge != nullptr) {
      result.push_back(curr_edge);
      curr_edge = edge(curr_edge->from_->idx_);
    }
    std::reverse(begin(result), end(result));
    return result;
  }

  inline size_t& dist(size_t const node_idx) const {
    return (*dists_)[node_idx - node_idx_offset_];
  }
  inline osm_edge const*& edge(size_t const node_idx) const {
    return (*edges_)[node_idx - node_idx_offset_];
  }

  inline size_t heuristic_distance(size_t const node_idx,
                                   size_t const initial_dist) const {
    auto const node_goal_dist = std::ceil(
        geo::haversine_distance(goal_center_, graph_.nodes_[node_idx]->xyz_));
    return initial_dist +
           (node_goal_dist > goal_radius_ ? node_goal_dist - goal_radius_ : 0);
  }

  osm_graph const& graph_;
  size_t node_idx_offset_;

  pq_t* pq_;
  std::vector<size_t>* dists_;
  std::vector<osm_edge const*>* edges_;

  std::vector<size_t> open_goals_;

  geo::xyz goal_center_{};
  size_t goal_radius_{0};

  size_t limit_{10};
};

std::vector<std::vector<osm_edge const*>> shortest_paths(
    osm_graph const& graph, size_t const& from, std::vector<size_t> const& to,
    bool ignore_limit) {

  osm_graph_dijkstra dijkstra{graph, from, to, ignore_limit};
  dijkstra.run();

  return utl::to_vec(to,
                     [&](auto const& id) { return dijkstra.get_edges(id); });
}

std::vector<double> shortest_path_distances(osm_graph const& graph,
                                            size_t const& from,
                                            std::vector<size_t> const& to) {
  std::vector<double> distances(to.size(),
                                std::numeric_limits<double>::infinity());

  auto const* from_node = graph.nodes_[from].get();

  std::vector<size_t> open_idx;
  for (auto i = 0UL; i < to.size(); ++i) {
    auto const* to_node = graph.nodes_[to[i]].get();

    if (from_node == to_node ||
        from_node->component_id_ != to_node->component_id_) {
      continue;
    }

    auto const it =
        std::find_if(begin(from_node->edges_), end(from_node->edges_),
                     [&](auto const& edge) { return edge.to_ == to_node; });
    if (it != end(from_node->edges_)) {
      distances[i] = it->dist_;
    } else {
      open_idx.push_back(i);
    }
  }

  if (!open_idx.empty()) {
    auto open_to = utl::to_vec(open_idx, [&](auto const i) { return to[i]; });

    osm_graph_dijkstra dijkstra{graph, from, open_to, false};
    dijkstra.run();

    for (auto i = 0UL; i < open_idx.size(); ++i) {
      auto dist = dijkstra.dist(open_to[i]);
      if (dist != std::numeric_limits<size_t>::max()) {
        distances[open_idx[i]] = dist;
      }
    }
  }

  return distances;
}

}  // namespace motis::path
