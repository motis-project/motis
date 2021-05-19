#include "motis/core/access/bfs.h"

#include <queue>

namespace motis {

std::set<trip::route_edge> route_bfs(ev_key const& k, bfs_direction const dir,
                                     bool with_through_edges) {
  std::set<trip::route_edge> visited;
  std::queue<trip::route_edge> q;

  visited.insert(k.route_edge_);
  q.push(k.route_edge_);

  while (!q.empty()) {
    auto const e = q.front().get_edge();
    q.pop();

    if (dir == bfs_direction::BOTH || dir == bfs_direction::BACKWARD) {
      for (auto const& in : e->from_->incoming_edges_) {
        if (in->type() != edge::THROUGH_EDGE &&
            in->type() != edge::ROUTE_EDGE) {
          continue;
        }

        if (visited.emplace(in).second) {
          q.push(trip::route_edge{in});
        }
      }
    }

    if (dir == bfs_direction::BOTH || dir == bfs_direction::FORWARD) {
      for (auto const& out : e->to_->edges_) {
        if (out.type() != edge::THROUGH_EDGE &&
            out.type() != edge::ROUTE_EDGE) {
          continue;
        }

        if (visited.insert(&out).second) {
          q.push(&out);
        }
      }
    }
  }

  if (!with_through_edges) {
    for (auto it = begin(visited); it != end(visited);) {
      if ((*it)->type() == edge::THROUGH_EDGE) {
        it = visited.erase(it);
      } else {
        ++it;
      }
    }
  }

  return visited;
}

std::set<ev_key> trip_bfs(ev_key const& k, bfs_direction const dir) {
  std::set<ev_key> ev_keys;
  for (auto const& e : route_bfs(k, dir)) {
    auto const arr = ev_key{e, k.lcon_idx_, event_type::ARR, k.day_};
    auto const dep = ev_key{e, k.lcon_idx_, event_type::DEP, k.day_};

    auto const bad_arr = dir == bfs_direction::BACKWARD && k.is_departure() &&
                         arr == k.get_opposite();
    if (!bad_arr) {
      ev_keys.insert(arr);
    }

    auto const bad_dep = dir == bfs_direction::FORWARD && k.is_arrival() &&
                         dep == k.get_opposite();
    if (!bad_dep) {
      ev_keys.insert(dep);
    }
  }
  return ev_keys;
}

}  // namespace motis
