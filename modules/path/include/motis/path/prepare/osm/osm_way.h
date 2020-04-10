#pragma once

#include <string>
#include <vector>

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

struct osm_way {
  osm_way(int64_t from, int64_t to, int64_t id, osm_path path, bool oneway)
      : from_(from),
        to_(to),
        ids_(std::vector<int64_t>{id}),
        path_(std::move(path)),
        oneway_(oneway) {}

  osm_way(int64_t from, int64_t to, std::vector<int64_t> ids, osm_path path,
          bool oneway)
      : from_(from),
        to_(to),
        ids_(std::move(ids)),
        path_(std::move(path)),
        oneway_(oneway) {}

  bool is_valid() const { return !ids_.empty(); }
  void invalidate() { ids_.clear(); }

  friend bool operator==(osm_way const& lhs, osm_way const& rhs) {
    return std::tie(lhs.from_, lhs.to_, lhs.ids_, lhs.path_.polyline_,
                    lhs.path_.osm_node_ids_, lhs.oneway_) ==
           std::tie(rhs.from_, rhs.to_, rhs.ids_, rhs.path_.polyline_,
                    rhs.path_.osm_node_ids_, rhs.oneway_);
  }

  int64_t from_, to_;

  std::vector<int64_t> ids_;
  osm_path path_;

  bool oneway_;
};

void aggregate_osm_ways(std::vector<osm_way>& osm_ways);

}  // namespace motis::path
