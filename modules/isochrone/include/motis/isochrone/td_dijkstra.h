#pragma once

#include <queue>

#include "boost/container/vector.hpp"
#include "utl/erase_if.h"

#include "motis/hash_map.h"

#include "motis/core/common/dial.h"
#include "build_query.h"
#include "motis/isochrone/statistics.h"

namespace motis::isochrone {

  const bool FORWARDING = true;

class td_dijkstra {
public:
  using dist_t = uint32_t;

  struct label {
    label(const node* const node, time now, bool last_conn_is_train) :
      node_(node),
      now_(now),
      last_conn_is_train_(last_conn_is_train){}

    friend bool operator>(label const& a, label const& b) {
      return a.now_ > b.now_;
    }

    const node* const node_;
    time now_;
    bool last_conn_is_train_;
  };

  struct get_bucket {
    std::size_t operator()(label const& l) const { return l.now_; }
  };

  enum : time { UNREACHABLE = std::numeric_limits<time>::max() };

  td_dijkstra (const node* const start, time begin_time, time end_time, const schedule* sched) :
  end_time_(end_time),
  sched_(sched) {
    times_.resize(2000000, UNREACHABLE);
    is_result_.resize(2000000, false);
    times_[start->id_] = begin_time;
    pq_.push(label(start, begin_time, false));
  }

  // dijkstra
  void run() {
    while (!pq_.empty()) {
      auto label = pq_.top();
      if(!is_result_[label.node_->get_station()->id_]
              && (label.node_->is_route_node() || (label.node_->is_station_node() && !label.last_conn_is_train_))) {
        results_.push_back(label);
        is_result_[label.node_->get_station()->id_] = true;
      }
      pq_.pop();

      for (auto const& edge : label.node_->edges_) {
        expand_edge(label.now_, edge, label.last_conn_is_train_);
      }
    }
  }

  inline void expand_edge(time t, edge const& edge, bool last_conn_is_train) {

    auto l = light_connection();
    light_connection const* lcon = nullptr;
    if(last_conn_is_train) {
      lcon = &l;
    }
    auto ec = edge.get_edge_cost(t, lcon);

    time new_time = (ec.time_<=end_time_) ? t + ec.time_ : UNREACHABLE;
    if (new_time < times_[edge.to_->id_] && new_time <= end_time_) {
      times_[edge.to_->id_] = new_time;
      pq_.push(label(edge.to_, new_time, edge.type() == edge::ROUTE_EDGE));
    }

  }

  statistics get_statistics() {
    return stats_;
  }


  std::vector<station*> get_stations() {
    auto stations = std::vector<station*>(results_.size());
    std::transform(results_.begin(), results_.end(), stations.begin(),
                   [this](label l) -> station* { return (sched_->stations_[l.node_->get_station()->id_]).get(); });
    return stations;
  }


  std::vector<long> get_remaining_times() {
    auto remaining_times = std::vector<long>(results_.size());
    std::transform(results_.begin(), results_.end(), remaining_times.begin(),
                   [this](label l) -> long { return end_time_-l.now_ ; });
    return remaining_times;
  }
private:
  dial<label, 12000, get_bucket> pq_;
  mcd::vector<time> times_;
  mcd::vector<bool> is_result_;
  time end_time_;
  std::vector<label> results_;
  const schedule* sched_;
  statistics stats_;
};

} // namespace motis::isochrone

