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
    label(const node* const node, time now) : node_(node), now_(now) {}

    friend bool operator>(label const& a, label const& b) {
      return a.now_ > b.now_;
    }

    const node* const node_;
    time now_;
  };

  struct get_bucket {
    std::size_t operator()(label const& l) const { return l.now_; }
  };

  enum : time { UNREACHABLE = std::numeric_limits<time>::max() };

  td_dijkstra (const node* const start, time begin_time, time end_time) :
  end_time_(end_time){
    times_.resize(2000000, UNREACHABLE);
    times_[start->id_] = begin_time;
    pq_.push(label(start, begin_time));
  }

  // dijkstra
  void run() {
    while (!pq_.empty()) {
      auto label = pq_.top();
      if(label.node_->is_station_node()) {
        results_.push_back(label);
      }
      pq_.pop();

      for (auto const& edge : label.node_->edges_) {
        expand_edge(label.now_, edge);
      }
    }
  }

  inline void expand_edge(time t, edge const& edge) {

    auto ec = edge.get_edge_cost(t, nullptr);

    time new_time = (ec.time_<=end_time_) ? t + ec.time_ : UNREACHABLE;
    if (new_time < times_[edge.to_->id_] && new_time <= end_time_) {
      times_[edge.to_->id_] = new_time;
      pq_.push(label(edge.to_, new_time));
    }

  }
private:
  dial<label, 12000, get_bucket> pq_;
  mcd::vector<time> times_;
  time end_time_;
  std::vector<label> results_;
};

} // namespace motis::isochrone

