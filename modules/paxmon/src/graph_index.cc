#include "motis/paxmon/graph_index.h"

#include "motis/paxmon/graph.h"

namespace motis::paxmon {

edge* edge_index::get(graph const& g) const {
  return &g.graph_.nodes_.at(node_).outgoing_edges(g).at(out_edge_idx_);
}

edge_index get_edge_index(graph const& g, edge const* e) {
  auto const node_idx = e->from_;
  for (auto const& [i, ep] :
       utl::enumerate(g.graph_.outgoing_edges(node_idx))) {
    if (&ep == e) {
      return edge_index{node_idx, static_cast<std::uint32_t>(i)};
    }
  }
  throw std::runtime_error{"get_edge_index: edge not found"};
}

}  // namespace motis::paxmon
