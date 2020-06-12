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

using node_id_t = uint32_t;

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

  mcd::indexed_vector<edge> edges_;
  mcd::vector<ptr<edge>> incoming_edges_;
  ptr<station_node> station_node_{nullptr};
  int32_t route_{-1};
  node_id_t id_{0};

  // Station Node Properties
  mcd::unique_ptr<node> foot_node_;
  mcd::vector<mcd::unique_ptr<node>> route_nodes_;
};

inline node make_node(node* station_node, node_id_t const node_id,
                      int32_t const route = -1) {
  node n;
  n.station_node_ = station_node;
  n.id_ = node_id;
  n.route_ = route;
  return n;
}

inline station_node make_station_node(node_id_t const id) {
  return make_node(nullptr, id);
}

using station_node_ptr = mcd::unique_ptr<station_node>;

}  // namespace motis
