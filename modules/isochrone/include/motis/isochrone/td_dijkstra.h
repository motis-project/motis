#pragma once

#include <queue>

#include "boost/container/vector.hpp"
#include "utl/erase_if.h"

#include "motis/hash_map.h"

#include "motis/core/common/dial.h"
#include "build_query.h"

namespace motis::isochrone {

  const bool FORWARDING = true;

class td_dijkstra {
public:
  using dist_t = uint32_t;

  struct label {
    label(const node* const node, uint32_t dist,  time now) : node_(node), dist_(dist), now_(now) {}

    friend bool operator>(label const& a, label const& b) {
      return a.dist_ > b.dist_;
    }

    const node* const node_;
    dist_t dist_;
    time now_;
  };

  struct get_bucket {
    std::size_t operator()(label const& l) const { return l.dist_; }
  };

  enum : dist_t { UNREACHABLE = std::numeric_limits<dist_t>::max() };

  td_dijkstra (const node* const start, time begin_time) :
  begin_time_(begin_time){
    dists_.resize(2000000, UNREACHABLE);
    dists_[start->id_] = 0;
    pq_.push(label(start, 0, begin_time));
  }

  // dijkstra
  void run() {
    while (!pq_.empty()) {
      auto label = pq_.top();
      pq_.pop();

      for (auto const& edge : label.node_->edges_) {
        expand_edge(label.dist_, edge);
      }
    }
  }

  inline void expand_edge(uint32_t dist, edge const& edge) {
    /*
    auto ec = edge.get_edge_cost(begin_time_, nullptr);
    uint32_t new_dist = dist + ec.time_;  // NOLINT
    if (new_dist < dists_[edge.to_] && new_dist <= MaxValue) {
      dists_[edge.to_->id_] = new_dist;
      pq_.push(label(edge.to_, new_dist));
    }
     */
  }
private:
  dial<label, std::numeric_limits<uint32_t>::max(), get_bucket> pq_;
  mcd::vector<dist_t> dists_;
  time begin_time_;
};

} // namespace motis::isochrone

