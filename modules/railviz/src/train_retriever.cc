#include "motis/railviz/train_retriever.h"

#include <algorithm>
#include <set>
#include <vector>

#include "boost/geometry/index/rtree.hpp"

#include "geo/detail/register_box.h"

#include "utl/get_or_create.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/bfs.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

namespace bgi = boost::geometry::index;

using value = std::pair<geo::box, std::pair<int, int>>;
using rtree = bgi::rtree<value, bgi::quadratic<16>>;

namespace motis::railviz {

bool is_relevant(edge const& e, int clasz) {
  return !e.empty() && e.m_.route_edge_.conns_[0].full_con_->clasz_ == clasz;
}

geo::latlng station_coords(schedule const& sched, unsigned station_idx) {
  auto const& station = sched.stations_[station_idx];
  return geo::latlng{station->lat(), station->lng()};
}

struct edge_geo_index {
  edge_geo_index(schedule const& sched, int clasz, rtree tree,
                 mcd::hash_set<std::pair<int, int>> included_station_pairs)
      : sched_{sched},
        clasz_{clasz},
        tree_{std::move(tree)},
        included_station_pairs_{std::move(included_station_pairs)} {}

  std::vector<edge const*> edges(geo::box const& b) const {
    std::vector<value> result_n;
    tree_.query(bgi::intersects(b), std::back_inserter(result_n));

    std::vector<edge const*> edges;
    for (auto const& result_pair : result_n) {
      resolve_edges(result_pair.second, edges);
    }
    return edges;
  }

  void resolve_edges(std::pair<int, int> const& stations,
                     std::vector<edge const*>& edges) const {
    auto const dir1 = std::make_pair(stations.second, stations.first);
    auto const dir2 = std::make_pair(stations.first, stations.second);

    for (auto const& d : {dir1, dir2}) {
      auto const from = sched_.station_nodes_[d.first].get();
      auto const to = sched_.station_nodes_[d.second].get();

      from->for_each_route_node([&](node const* route_node) {
        for (auto const& e : route_node->edges_) {
          if (!is_relevant(e, clasz_) || e.to_->get_station() != to) {
            continue;
          }

          edges.emplace_back(&e);
        }
      });
    }
  }

  schedule const& sched_;
  int clasz_;
  rtree tree_;
  mcd::hash_set<std::pair<int, int>> included_station_pairs_;
};

std::unique_ptr<edge_geo_index> make_edge_rtree(
    schedule const& sched, int clasz,
    mcd::hash_map<std::pair<int, int>, geo::box> const& boxes) {
  std::vector<value> entries;
  mcd::hash_set<std::pair<int, int>> included_station_pairs;

  size_t no_match = 0;
  auto const get_bounding_box = [&](std::pair<int, int> const& stations) {
    auto const it = boxes.find(stations);
    no_match += (it == end(boxes) ? 1 : 0);
    return it == end(boxes)
               ? geo::make_box({station_coords(sched, stations.first),
                                station_coords(sched, stations.second)})
               : it->second;
  };

  auto const add_edges_of_route_node = [&](node const* route_node) {
    assert(route_node->is_route_node());
    for (auto const& e : route_node->edges_) {
      if (!is_relevant(e, clasz)) {
        continue;
      }

      auto const from = e.from_->get_station()->id_;
      auto const to = e.to_->get_station()->id_;
      std::pair<int, int> const station_pair(std::min(from, to),
                                             std::max(from, to));
      if (!included_station_pairs.insert(station_pair).second) {
        continue;
      }

      entries.emplace_back(get_bounding_box(station_pair), station_pair);
    }
  };

  auto const add_edges_of_station = [&](station_node const* sn) {
    assert(sn->is_station_node());
    sn->for_each_route_node(
        [&](node const* route_node) { add_edges_of_route_node(route_node); });
  };

  for (const auto& node : sched.station_nodes_) {
    add_edges_of_station(node.get());
  }

  if (no_match != 0) {
    LOG(logging::warn) << "clasz " << clasz
                       << ": station pairs without tbbox: " << no_match << "/"
                       << entries.size();
  }

  return std::make_unique<edge_geo_index>(sched, clasz, rtree{entries},
                                          std::move(included_station_pairs));
}

constexpr auto const RELEVANT_CLASSES = NUM_CLASSES;

train_retriever::train_retriever(
    schedule const& sched,
    mcd::hash_map<std::pair<int, int>, geo::box> const& boxes)
    : sched_{sched} {
  edge_index_.resize(RELEVANT_CLASSES);
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_[clasz] = make_edge_rtree(sched, clasz, boxes);
  }
}

train_retriever::~train_retriever() = default;

void train_retriever::update(rt::RtUpdates const* updates) {
  std::unique_lock lock(mutex_);

  std::vector<std::vector<value>> new_values;
  new_values.resize(RELEVANT_CLASSES);

  for (auto const* update : *updates->updates()) {
    if (update->content_type() != rt::Content_RtRerouteUpdate) {
      continue;
    }

    auto const reroute_update =
        reinterpret_cast<rt::RtRerouteUpdate const*>(update->content());
    for (auto const& section :
         access::sections(from_fbs(sched_, reroute_update->trip()))) {
      std::pair<int, int> const station_pair(
          std::min(section.from_station_id(), section.to_station_id()),
          std::max(section.from_station_id(), section.to_station_id()));

      auto const clasz = section.fcon().clasz_;
      if (!edge_index_.at(clasz)
               ->included_station_pairs_.insert(station_pair)
               .second) {
        continue;
      }

      new_values.at(clasz).emplace_back(
          geo::make_box({station_coords(sched_, station_pair.first),
                         station_coords(sched_, station_pair.second)}),
          station_pair);
    }
  }

  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_.at(clasz)->tree_.insert(new_values.at(clasz));
  }
}

bool should_display(int clasz, int zoom_level, float distance = 0.F) {
  return clasz < 3  //
         || (clasz < 6 && zoom_level >= 4)  //
         || (clasz < 7 && zoom_level >= 6)  //
         || (clasz >= 7 && zoom_level == 10 && distance >= 10'000.F)  //
         || zoom_level > 10;
}

std::vector<train> train_retriever::trains(time const start_time,
                                           time const end_time,
                                           unsigned const max_count,
                                           geo::box const& area,
                                           int zoom_level) {
  mcd::hash_map<node const*, float> route_distances;
  auto const get_or_create_route_distance = [&](ev_key const& k) {
    auto const it =
        route_distances.find(cista::ptr_cast(k.route_edge_.route_node_));
    if (it != end(route_distances)) {
      return it->second;
    }

    auto const route_edges = route_bfs(k, bfs_direction::BOTH, false);

    geo::box b;
    for (auto const& re : route_edges) {
      auto const* e = re.get_edge();

      auto const& s_dep = sched_.stations_.at(e->from_->get_station()->id_);
      b.extend({s_dep->lat(), s_dep->lng()});

      auto const& s_arr = sched_.stations_.at(e->to_->get_station()->id_);
      b.extend({s_arr->lat(), s_arr->lng()});
    }

    float const distance = geo::distance(b.min_, b.max_);
    for (auto const& re : route_edges) {
      route_distances[cista::ptr_cast(re.route_node_)] = distance;
    }
    return distance;
  };

  constexpr auto const kBusClasz = 8;

  std::shared_lock lock(mutex_);
  std::vector<train> trains;
  for (auto clasz = 0U; clasz < kBusClasz; ++clasz) {
    if (!should_display(clasz, zoom_level)) {
      continue;
    }

    for (auto const& e : edge_index_[clasz]->edges(area)) {
      for (auto i = 0U; i < e->m_.route_edge_.conns_.size(); ++i) {
        auto const& c = e->m_.route_edge_.conns_[i];
        if (c.valid_ == 0U || c.a_time_ < start_time || c.d_time_ > end_time) {
          continue;
        }

        auto const k = ev_key{e, i, event_type::DEP};
        float const distance = get_or_create_route_distance(k);
        trains.emplace_back(train{k, distance});

        if (trains.size() >= max_count) {
          goto max_count_reached;
        }
      }
    }
  }

  if (should_display(kBusClasz, zoom_level,
                     std::numeric_limits<float>::infinity())) {
    std::vector<train> busses;
    for (auto const& e : edge_index_[kBusClasz]->edges(area)) {
      for (auto i = 0U; i < e->m_.route_edge_.conns_.size(); ++i) {
        auto const& c = e->m_.route_edge_.conns_[i];
        if (c.valid_ == 0U || c.a_time_ < start_time || c.d_time_ > end_time) {
          continue;
        }

        auto const k = ev_key{e, i, event_type::DEP};
        auto const distance = get_or_create_route_distance(k);
        if (!should_display(kBusClasz, zoom_level, distance)) {
          continue;
        }

        busses.push_back({k, distance});
      }
    }

    // .distance DESC, .key ASC
    std::sort(begin(busses), end(busses), [](auto const& lhs, auto const& rhs) {
      return std::tie(rhs.route_distance_, lhs.key_) <
             std::tie(lhs.route_distance_, rhs.key_);
    });

    for (auto const& b : busses) {
      if (trains.size() >= max_count) {
        goto max_count_reached;
      }
      trains.push_back(b);
    }
  }

max_count_reached:
  return trains;
}

}  // namespace motis::railviz
