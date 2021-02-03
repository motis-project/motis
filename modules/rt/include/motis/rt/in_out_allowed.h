#pragma once

#include <map>

#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"

namespace motis::rt {

struct in_out_allowed {
  in_out_allowed() = default;
  in_out_allowed(bool in_allowed, bool out_allowed)
      : in_allowed_(in_allowed), out_allowed_(out_allowed) {}
  bool operator==(in_out_allowed const& o) const {
    return in_allowed_ == o.in_allowed_ && out_allowed_ == o.out_allowed_;
  }
  bool in_allowed_, out_allowed_;
};

inline bool get_in_allowed(node const* n) {
  return std::any_of(
      begin(n->incoming_edges_), end(n->incoming_edges_), [&](auto&& e) {
        return e->from_ == n->get_station() && e->type() != edge::INVALID_EDGE;
      });
}

inline bool get_out_allowed(node const* n) {
  return std::any_of(begin(n->edges_), end(n->edges_), [&](auto&& e) {
    return e.get_destination() == n->get_station() &&
           e.type() != edge::INVALID_EDGE;
  });
}

inline in_out_allowed get_in_out_allowed(node const* n) {
  return {get_in_allowed(n), get_out_allowed(n)};
}

}  // namespace motis::rt
