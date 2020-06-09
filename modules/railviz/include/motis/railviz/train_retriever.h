#pragma once

#include <shared_mutex>
#include <vector>

#include "geo/box.h"

#include "motis/hash_map.h"

#include "motis/core/schedule/event.h"
#include "motis/core/schedule/time.h"

#include "motis/railviz/train.h"

#include "motis/protocol/RtUpdate_generated.h"

namespace motis {

struct schedule;

namespace railviz {

struct edge_geo_index;

struct train_retriever {
  train_retriever(schedule const&,
                  mcd::hash_map<std::pair<int, int>, geo::box> const&);
  ~train_retriever();

  train_retriever(train_retriever const&) = delete;
  train_retriever& operator=(train_retriever const&) = delete;

  train_retriever(train_retriever&&) = delete;
  train_retriever& operator=(train_retriever&&) = delete;

  void update(rt::RtUpdates const*);
  std::vector<train> trains(time start_time, time end_time, int max_count,
                            int last_count, geo::box const& area,
                            int zoom_level);

private:
  schedule const& sched_;
  std::shared_mutex mutable mutex_;
  std::vector<std::unique_ptr<edge_geo_index>> edge_index_;
};

}  // namespace railviz
}  // namespace motis
