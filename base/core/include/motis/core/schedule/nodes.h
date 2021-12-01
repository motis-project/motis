#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/time.h"

namespace motis {

enum class node_type : uint8_t {
  STATION_NODE,
  ROUTE_NODE,
  FOOT_NODE,
  PLATFORM_NODE
};

struct node;

using station_node = node;

using node_id_t = uint32_t;

struct node {
  inline bool is_station_node() const {
    return type_ == node_type::STATION_NODE;
  }
  inline bool is_route_node() const { return type_ == node_type::ROUTE_NODE; }
  inline bool is_foot_node() const { return type_ == node_type::FOOT_NODE; }
  inline bool is_platform_node() const {
    return type_ == node_type::PLATFORM_NODE;
  }

  inline node_type type() const { return type_; }

  char const* type_str() const {
    switch (type_) {
      case node_type::STATION_NODE: return "STATION_NODE";
      case node_type::ROUTE_NODE: return "ROUTE_NODE";
      case node_type::FOOT_NODE: return "FOOT_NODE";
      case node_type::PLATFORM_NODE: return "PLATFORM_NODE";
      default: return "UNKNOWN_NODE";
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

  bool is_in_allowed() const {
    assert(is_route_node());
    return std::any_of(
        begin(incoming_edges_), end(incoming_edges_), [&](auto const& e) {
          return e->from_ == station_node_ && e->type() != edge::INVALID_EDGE;
        });
  }

  bool is_out_allowed() const {
    assert(is_route_node());
    return std::any_of(begin(edges_), end(edges_), [&](auto const& e) {
      return e.to_ == station_node_ && e.type() != edge::INVALID_EDGE;
    });
  }

  mcd::indexed_vector<edge> edges_;
  mcd::vector<ptr<edge>> incoming_edges_;
  ptr<station_node> station_node_{nullptr};
  int32_t route_{-1};
  node_id_t id_{0};
  node_type type_{node_type::STATION_NODE};

  // Station Node Properties
  mcd::unique_ptr<node> foot_node_;
  mcd::vector<mcd::unique_ptr<node>> child_nodes_;
  mcd::vector<node*> platform_nodes_;
};

inline node make_node(node_type const type, node* station_node,
                      node_id_t const node_id, int32_t const route = -1) {
  node n;
  n.type_ = type;
  n.station_node_ = station_node;
  n.id_ = node_id;
  n.route_ = route;
  return n;
}

inline station_node make_station_node(node_id_t const id) {
  return make_node(node_type::STATION_NODE, nullptr, id);
}

using station_node_ptr = mcd::unique_ptr<station_node>;

}  // namespace motis
