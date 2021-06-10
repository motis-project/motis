#pragma once

#include <queue>

#include "boost/container/vector.hpp"
#include "utl/erase_if.h"

#include "motis/hash_map.h"

#include "motis/core/common/dial.h"

#include "motis/isochrone/mem_manager.h"
#include "motis/isochrone/statistics.h"
#include "build_query.h"

namespace motis::isochrone {

const bool FORWARDING = true;

using td_graph = mcd::vector<mcd::vector<edge>>;

inline td_graph build_graph(mcd::vector<station_node_ptr> const& station_nodes) {
    td_graph g(station_nodes.size());

    return g;
}

template <search_dir Dir, typename Label, typename LowerBounds>
struct td_dijkstra {



  td_dijkstra(
      int node_count, unsigned int station_node_count,
      mcd::hash_map<node const*, std::vector<edge>> additional_edges,
      time max_time)
      : station_node_count_(station_node_count),
        additional_edges_(std::move(additional_edges)),
        max_time_(max_time)
        {}


  }

  void search() {

  }

  statistics get_statistics() const { return stats_; };


private:

  unsigned int station_node_count_;
  dial<Label*, Label::MAX_BUCKET, get_bucket> queue_;
  mcd::hash_map<node const*, std::vector<edge>> additional_edges_;
  time max_time_;
  statistics stats_;
  std::size_t max_labels_;
};

}  // namespace motis::isochrone
