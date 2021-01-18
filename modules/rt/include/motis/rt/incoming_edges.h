#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "utl/verify.h"

#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/trip.h"

namespace motis::rt {

struct incoming_edge_patch {
  incoming_edge_patch() = default;
  explicit incoming_edge_patch(edge const* e, int incoming_edge_idx = -1)
      : edge_(e), incoming_edge_idx_(incoming_edge_idx) {}

  trip::route_edge edge_;
  int incoming_edge_idx_{-1};
};

inline bool station_contains_node(station_node const* s, node const* n) {
  return n->get_station() == s;
}

inline void save_outgoing_edges(node const* n,
                                std::vector<incoming_edge_patch>& incoming) {
  for (auto const& e : n->edges_) {
    auto const incoming_edge = std::find(begin(e.to_->incoming_edges_),
                                         end(e.to_->incoming_edges_), &e);
    utl::verify(incoming_edge != end(e.to_->incoming_edges_),
                "incoming edge not found");
    auto const incoming_edge_idx =
        std::distance(begin(e.to_->incoming_edges_), incoming_edge);
    incoming.emplace_back(&e, static_cast<int>(incoming_edge_idx));
  }
}

inline void save_outgoing_edges(std::set<station_node*> const& station_nodes,
                                std::vector<incoming_edge_patch>& incoming) {
  for (auto const& s : station_nodes) {
    save_outgoing_edges(s, incoming);
    if (s->foot_node_) {
      save_outgoing_edges(s->foot_node_.get(), incoming);
    }
    for (auto const& pn : s->platform_nodes_) {
      if (pn != nullptr) {
        save_outgoing_edges(pn, incoming);
      }
    }
  }
}

inline void add_outgoing_edge(edge const* e,
                              std::vector<incoming_edge_patch>& incoming) {
  incoming.emplace_back(e);
}

inline void add_outgoing_edges(node const* n,
                               std::vector<incoming_edge_patch>& incoming) {
  for (auto const& e : n->edges_) {
    add_outgoing_edge(&e, incoming);
  }
}

inline void add_outgoing_edges_from_new_route(
    std::map<node const*, node*> const& nodes,
    std::vector<incoming_edge_patch>& incoming) {
  for (auto const& n : nodes) {
    add_outgoing_edges(n.second, incoming);
  }
}

inline void apply_incoming_edge_patch(incoming_edge_patch const& patch) {
  utl::verify(
      patch.edge_.outgoing_edge_idx_ < patch.edge_.route_node_->edges_.size(),
      "invalid route_edge");
  if (patch.incoming_edge_idx_ != -1) {
    utl::verify(static_cast<std::size_t>(patch.incoming_edge_idx_) <
                    patch.edge_->to_->incoming_edges_.size(),
                "invalid incoming edge index");
    patch.edge_->to_
        ->incoming_edges_[static_cast<std::size_t>(patch.incoming_edge_idx_)] =
        patch.edge_.get_edge();
  } else {
    patch.edge_->to_->incoming_edges_.push_back(patch.edge_.get_edge());
  }
}

inline void patch_incoming_edges(
    std::vector<incoming_edge_patch> const& incoming) {
  for (auto const& patch : incoming) {
    apply_incoming_edge_patch(patch);
  }
}

}  // namespace motis::rt
