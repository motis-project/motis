#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/event.h"
#include "motis/core/schedule/time.h"

#include "motis/railviz/geo.h"

#include "motis/protocol/PathBoxesResponse_generated.h"

namespace motis {

struct schedule;

namespace railviz {

struct edge_geo_index;

struct train_retriever {
  train_retriever(schedule const& s,
                  mcd::hash_map<std::pair<int, int>, geo::box> const&);
  ~train_retriever();

  train_retriever(train_retriever const&) = delete;
  train_retriever& operator=(train_retriever const&) = delete;

  train_retriever(train_retriever&&) = default;
  train_retriever& operator=(train_retriever&&) = default;

  std::vector<ev_key> trains(time from, time to, unsigned max_count,
                             geo::box const& area);

private:
  std::vector<std::unique_ptr<edge_geo_index>> edge_index_;
};

}  // namespace railviz
}  // namespace motis
