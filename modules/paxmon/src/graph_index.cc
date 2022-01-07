#include "motis/paxmon/graph_index.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

edge* edge_index::get(universe const& uv) const {
  return &uv.graph_.nodes_.at(node_).outgoing_edges(uv).at(out_edge_idx_);
}

edge_index get_edge_index(universe const& uv, edge const* e) {
  auto const node_idx = e->from();
  for (auto const& [i, ep] :
       utl::enumerate(uv.graph_.outgoing_edges(node_idx))) {
    if (&ep == e) {
      return edge_index{node_idx, static_cast<std::uint32_t>(i)};
    }
  }
  throw std::runtime_error{"get_edge_index: edge not found"};
}

}  // namespace motis::paxmon
