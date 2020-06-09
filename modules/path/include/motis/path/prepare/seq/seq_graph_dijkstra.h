#pragma once

#include <queue>

#include "utl/erase.h"

namespace motis::path {

// TODO (sebastian) implement same perf improvements as with osm_router!?
struct seq_graph_dijkstra {
  struct label {
    label(size_t const idx, size_t const dist, seq_edge const* link)
        : idx_(idx), dist_(dist), link_(link) {}

    friend bool operator>(label const& a, label const& b) {
      return a.dist_ > b.dist_;
    }

    size_t idx_, dist_;
    seq_edge const* link_;
  };

  seq_graph_dijkstra(seq_graph const& graph, std::vector<size_t> const& initial,
                     std::vector<size_t> const& goals)
      : graph_(graph), goals_(goals), open_goals_(goals) {

    dists_.resize(graph_.nodes_.size(), std::numeric_limits<size_t>::max());
    links_.resize(graph_.nodes_.size(), nullptr);

    for (auto const& i : initial) {
      dists_[i] = 0;
      pq_.push(label(i, 0, nullptr));
    }
  }

  void run() {

    while (!pq_.empty()) {
      auto label = pq_.top();
      pq_.pop();

      auto const this_idx = label.idx_;

      utl::erase(open_goals_, this_idx);
      if (open_goals_.empty()) {
        break;
      }

      auto const& node = graph_.nodes_[this_idx];
      for (auto const& link : node->edges_) {
        size_t const new_dist = label.dist_ + link.weight();
        size_t const to_idx = link.to_->idx_;
        if (new_dist < dists_[to_idx]) {
          dists_[to_idx] = new_dist;
          links_[to_idx] = &link;
          pq_.push({to_idx, new_dist, &link});
        }
      }
    }
  }

  size_t get_best_goal() const {
    utl::verify(!graph_.goals_.empty(), "get_best_goal: INVALID_GRAPH");
    return *std::min_element(begin(graph_.goals_), end(graph_.goals_),
                             [&, this](auto const& lhs, auto const& rhs) {
                               return this->get_distance(lhs) <
                                      this->get_distance(rhs);
                             });
  }

  std::vector<seq_edge const*> get_links(size_t const goal) const {
    std::vector<seq_edge const*> result;

    auto link = links_[goal];
    while (link != nullptr) {
      result.push_back(link);
      link = links_[link->from_->idx_];
    }

    std::reverse(begin(result), end(result));
    return result;
  }

  size_t get_distance(size_t const goal) const { return dists_[goal]; }

  seq_graph const& graph_;
  std::priority_queue<label, std::vector<label>, std::greater<>> pq_;

  std::vector<size_t> dists_;
  std::vector<seq_edge const*> links_;

  std::vector<size_t> goals_;
  std::vector<size_t> open_goals_;
};

inline std::vector<seq_edge const*> find_shortest_path(seq_graph const& graph) {
  seq_graph_dijkstra dijkstra(graph, graph.initials_, graph.goals_);
  dijkstra.run();

  auto const best_goal = dijkstra.get_best_goal();
  return dijkstra.get_links(best_goal);
}

}  // namespace motis::path
