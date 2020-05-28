#include "motis/railviz/train_retriever.h"

#include "motis/core/schedule/schedule.h"

#include "motis/railviz/edge_geo_index.h"

namespace motis::railviz {

constexpr auto const RELEVANT_CLASSES = NUM_CLASSES;

train_retriever::train_retriever(
    schedule const& s,
    mcd::hash_map<std::pair<int, int>, geo::box> const& boxes) {
  edge_index_.resize(RELEVANT_CLASSES);
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_[clasz] = std::make_unique<edge_geo_index>(clasz, s, boxes);
  }
}

train_retriever::~train_retriever() = default;

std::vector<ev_key> train_retriever::trains(time const from, time const to,
                                            unsigned const max_count,
                                            geo::box const& area) {
  std::vector<ev_key> connections;
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    for (auto const& e : edge_index_[clasz]->edges(area)) {
      for (auto i = 0U; i < e->m_.route_edge_.conns_.size(); ++i) {
        auto const& con = e->m_.route_edge_.conns_[i];
        if (con.a_time_ >= from && con.d_time_ <= to && (con.valid_ != 0U)) {
          connections.emplace_back(ev_key{e, i, event_type::DEP});
          if (connections.size() >= max_count) {
            goto end;
          }
        }
      }
    }
  }
end:
  return connections;
}

}  // namespace motis::railviz
