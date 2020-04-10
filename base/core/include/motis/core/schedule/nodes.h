#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/time.h"

namespace motis {

enum class node_type { STATION_NODE, ROUTE_NODE, FOOT_NODE };

struct node;

using station_node = node;

struct node {
  bool is_station_node() const { return station_node_ == nullptr; }
  bool is_route_node() const { return route_ != -1; }
  bool is_foot_node() const { return !is_station_node() && !is_route_node(); }

  node_type type() const {
    if (is_station_node()) {
      return node_type::STATION_NODE;
    } else if (is_route_node()) {
      return node_type::ROUTE_NODE;
    } else {
      return node_type::FOOT_NODE;
    }
  }

  char const* type_str() const {
    if (is_station_node()) {
      return "STATION_NODE";
    } else if (is_route_node()) {
      return "ROUTE_NODE";
    } else {
      return "FOOT_NODE";
    }
  }

  station_node* as_station_node() {
    if (station_node_ == nullptr) {
      return reinterpret_cast<station_node*>(this);
    } else {
      return nullptr;
    }
  }

  station_node const* as_station_node() const {
    if (station_node_ == nullptr) {
      return reinterpret_cast<station_node const*>(this);
    } else {
      return nullptr;
    }
  }

  station_node* get_station() {
    if (station_node_ == nullptr) {
      return reinterpret_cast<station_node*>(this);
    } else {
      return station_node_;
    }
  }

  station_node const* get_station() const {
    if (station_node_ == nullptr) {
      return reinterpret_cast<station_node const*>(this);
    } else {
      return station_node_;
    }
  }
  template <typename Fn>
  void for_each_route_node(Fn&& f) const {
    for (auto& edge : edges_) {
      if (edge.to_->is_route_node()) {
        f(edge.to_);
      }
    }
  }

  int add_foot_node(int node_id) {
    if (!foot_node_) {
      foot_node_ = mcd::make_unique<node>();
      foot_node_->station_node_ = this;
      foot_node_->id_ = node_id++;
      for_each_route_node([&](auto&& route_node) {
        // check whether it is allowed to transfer at the route-node
        // we do this by checking, whether it has an edge to the station
        for (auto const& edge : route_node->edges_) {
          if (edge.get_destination() == this &&
              edge.type() != edge::INVALID_EDGE) {
            // the foot-edge may only be used
            // if a train was used beforewards when
            // trying to use it from a route node
            route_node->edges_.push_back(make_after_train_fwd_edge(
                route_node, foot_node_.get(), 0, true));
            break;
          }
        }
      });
      for (auto const& edge : edges_) {
        if (edge.get_destination()->is_route_node() &&
            edge.type() != edge::INVALID_EDGE) {
          foot_node_->edges_.emplace_back(
              make_after_train_bwd_edge(foot_node_.get(), edge.to_, 0, true));
        }
      }
      edges_.emplace_back(make_fwd_edge(this, foot_node_.get()));
      foot_node_->edges_.emplace_back(make_bwd_edge(foot_node_.get(), this));
    }
    return node_id;
  }

  int add_foot_edge(int node_id, edge fe) {
    node_id = add_foot_node(node_id);
    node_id = fe.to_->get_station()->add_foot_node(node_id);
    fe.from_ = foot_node_.get();
    foot_node_->edges_.emplace_back(fe);
    edges_.emplace_back(
        make_bwd_edge(this, fe.to_->get_station()->foot_node_.get(),
                      fe.m_.foot_edge_.time_cost_, fe.m_.foot_edge_.transfer_));
    return node_id;
  }

  mcd::indexed_vector<edge> edges_;
  mcd::vector<ptr<edge>> incoming_edges_;
  ptr<station_node> station_node_{nullptr};
  int32_t route_{-1};
  uint32_t id_{0};

  // Station Node Properties
  mcd::unique_ptr<node> foot_node_;
  mcd::vector<mcd::unique_ptr<node>> route_nodes_;
};

inline node make_node(node* station_node, int node_id, int32_t route = -1) {
  node n;
  n.station_node_ = station_node;
  n.id_ = node_id;
  n.route_ = route;
  return n;
}

inline station_node make_station_node(unsigned id) {
  return make_node(nullptr, id);
}

using station_node_ptr = mcd::unique_ptr<station_node>;

}  // namespace motis
