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
    label(const node* node, time now, edge const* pred_edge) :
      node_(node),
      now_(now),
      pred_edge(pred_edge){}

    friend bool operator>(label const& a, label const& b) {
      return a.now_ > b.now_;
    }


    const node* node_;
    time now_;
    edge const* pred_edge;
  };

  struct get_bucket {
    std::size_t operator()(label const& l) const { return l.now_; }
  };

  enum : time { UNREACHABLE = std::numeric_limits<time>::max() };

  void generate_start_labels(const std::vector<std::pair<const node *, time>> &starts) {
    //todo: generate start labels with dijkstra
    dial<label, 12000, get_bucket> pq;
    std::set<uint32_t> visited;
    for(auto start : starts) {
      times_[start.first->id_] = start.second;
      pq.push(label(start.first,  start.second, nullptr));
    }
    while(!pq.empty()) {
      auto l = pq.top();
      pq.pop();
      if(!l.node_->is_station_node() && visited.find(l.node_->id_) != end(visited)) {
        continue;
      }
      for(auto const &edge : l.node_->edges_) {
        if(edge.type() == edge::ENTER_EDGE) {
          auto time = l.now_;

          if(l.pred_edge == nullptr) {
            time += sched_->stations_[l.node_->id_]->transfer_time_;
          }
          pq.push(label(edge.get_destination(), time, &edge));
        } else if (edge.type() == edge::FWD_EDGE) {
          pq.push(label(edge.get_destination(), l.now_+edge.m_.foot_edge_.time_cost_, &edge));
        }
      }
      visited.insert(l.node_->id_);
      if(l.node_->is_route_node()) {
        pq_.push(l);
      }
    }
    /*
    for(auto start :starts) {
      times_[start.first->id_] = start.second;
      for(const auto& edge : start.first->edges_) {
        if(edge.type() == edge::ENTER_EDGE) {
          auto start_time = start.second + sched_->stations_[start.first->id_]->transfer_time_;
          pq_.push(label(edge.get_destination(), start_time, &edge));
          times_[edge.get_destination()->id_] = start_time;
        }
      }
    }*/
  }

  td_dijkstra (const std::vector<std::pair<const node*, time>>& starts, time begin_time, time end_time, const schedule* sched) :
  end_time_(end_time),
  sched_(sched) {
    times_.resize(2000000, UNREACHABLE);
    prev_.resize(2000000, -1);
    route_to_station_.resize(2000000, -1);
    end_times_.resize(sched_->stations_.size(), UNREACHABLE);
    is_result_.resize(sched_->stations_.size(), false);
    is_direct_.resize(sched_->stations_.size(), false);
    generate_start_labels(starts);
  }

  // dijkstra
  void run() {
    while (!pq_.empty()) {
      auto l = pq_.top();
      route_to_station_[l.node_->id_] = l.node_->get_station()->id_;
      if(!is_result_[l.node_->get_station()->id_]&&
          ((l.node_->is_route_node() && is_exit_possible(&l)))) {
        if(l.now_ > times_[l.node_->get_station()->id_]) {
          auto direct = label(l.node_->get_station(), times_[l.node_->get_station()->id_], nullptr);
          results_.push_back(direct);
          is_direct_[l.node_->get_station()->id_]=true;
        } else {
          results_.push_back(l);
        }
        is_result_[l.node_->get_station()->id_] = true;

      }
      pq_.pop();

      for (auto const& edge : l.node_->edges_) {
        expand_edge(&l, edge);
      }
    }
    meta_station_time_adjust();
  }

  inline void expand_edge(label* l, edge const& edge) {
    light_connection*  lcon;
    if(l->pred_edge->type()!=edge::ROUTE_EDGE) {
      lcon = nullptr;
    }
    auto ec = edge.get_edge_cost(l->now_, lcon);
    if(!ec.is_valid() || edge.get_destination() == l->pred_edge->get_source()) {
      return;
    }

    time new_time = (ec.time_<=end_time_) ? l->now_ + ec.time_ : UNREACHABLE;
    if (new_time < times_[edge.to_->id_] && new_time <= end_time_) {
      times_[edge.to_->id_] = new_time;
      pq_.push(label(edge.to_, new_time, &edge));
      prev_[edge.to_->id_] = edge.from_->id_;
    }

  }

  bool is_exit_possible(label* l){
    bool possible = false;
    for(auto const& edge : l->node_->edges_){
      if(edge.to_->is_station_node()) {
        possible = edge.valid();
      }
    }
    return possible;
  }

  void meta_station_time_adjust(){
    std::vector<label> orig_results(results_);
    for(size_t i = 0; i < orig_results.size(); ++i){
      auto const& meta_stations = sched_->stations_[orig_results[i].node_->get_station()->id_]->equivalent_;
      auto best_time = orig_results[i].now_;
      for(auto const &station : meta_stations) {
        for(auto & orig_result : orig_results) {
          if (orig_result.node_->get_station()->id_ == station->index_ && orig_result.now_ < best_time && !is_direct_[station->index_]) {
            best_time = orig_result.now_;
          }
        }
      }
      if(best_time < orig_results[i].now_) {
        results_[i].now_ = UINT16_MAX;
      }
    }

    for (auto it = results_.begin(); it != results_.end(); ) {
      if (it->now_ == UINT16_MAX) { it = results_.erase(it); }
      else                 { ++it;          }
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
                   [this](label l) -> long { return (end_time_-l.now_)*60 ; });
    return remaining_times;
  }

  void print_nodes(uint32_t node_id) {
    std::string str;
    auto cur_id = node_id;
    while(cur_id!=-1) {
      str = std::to_string(cur_id) + " "+sched_->stations_[route_to_station_[cur_id]]->name_.str()+ "\n" + str;
      cur_id = prev_[cur_id];
    }
    std::cout << str;
    std::cout << std::endl;
  }
private:
  dial<label, 12000, get_bucket> pq_;
  mcd::vector<time> times_;
  mcd::vector<time> end_times_;
  mcd::vector<uint32_t> prev_;
  mcd::vector<uint32_t> route_to_station_;
  mcd::vector<bool> is_result_;
  mcd::vector<bool> is_direct_;
  time end_time_;
  std::vector<label> results_;
  const schedule* sched_;
  statistics stats_;
};

} // namespace motis::isochrone

