#include "motis/railviz/train_retriever.h"

#include <algorithm>
#include <set>
#include <vector>

#include "boost/geometry/index/rtree.hpp"

#include "fmt/core.h"

#include "geo/detail/register_box.h"

#include "utl/concat.h"
#include "utl/get_or_create.h"
#include "utl/raii.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/bfs.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/path/path_zoom_level.h"

namespace bgi = boost::geometry::index;

using value = std::pair<geo::box, std::pair<int, int>>;
using rtree = bgi::rtree<value, bgi::quadratic<16>>;

namespace motis::railviz {

bool is_relevant(edge const& e, service_class clasz) {
  return !e.empty() && e.m_.route_edge_.conns_[0].full_con_->clasz_ == clasz;
}

geo::latlng station_coords(schedule const& sched, unsigned station_idx) {
  auto const& station = sched.stations_[station_idx];
  return geo::latlng{station->lat(), station->lng()};
}

struct edge_geo_index {
  edge_geo_index(schedule const& sched, service_class clasz, rtree tree,
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
  service_class clasz_;
  rtree tree_;
  mcd::hash_set<std::pair<int, int>> included_station_pairs_;
};

std::unique_ptr<edge_geo_index> make_edge_rtree(
    schedule const& sched, service_class clasz,
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
    LOG(logging::warn) << fmt::format(
        "clasz {} : stations without tbbox {} / {}", clasz, no_match,
        entries.size());
  }

  return std::make_unique<edge_geo_index>(sched, clasz, rtree{entries},
                                          std::move(included_station_pairs));
}

constexpr auto const RELEVANT_CLASSES =
    static_cast<service_class_t>(service_class::NUM_CLASSES);

train_retriever::train_retriever(
    schedule const& sched,
    mcd::hash_map<std::pair<int, int>, geo::box> const& boxes)
    : sched_{sched} {
  edge_index_.resize(RELEVANT_CLASSES);
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_[clasz] =
        make_edge_rtree(sched, static_cast<service_class>(clasz), boxes);
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
      if (!edge_index_.at(static_cast<service_class_t>(clasz))
               ->included_station_pairs_.insert(station_pair)
               .second) {
        continue;
      }

      new_values.at(static_cast<service_class_t>(clasz))
          .emplace_back(
              geo::make_box({station_coords(sched_, station_pair.first),
                             station_coords(sched_, station_pair.second)}),
              station_pair);
    }
  }

  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_.at(clasz)->tree_.insert(new_values.at(clasz));
  }
}

std::vector<train> train_retriever::trains(
    time const start_time, time const end_time, int const max_count,
    int const last_count, geo::box const& area, int const zoom_level) {
  constexpr auto const kTolerance = .1F;
  auto const limit = (last_count > max_count ||
                      std::abs(last_count - max_count) < max_count * kTolerance)
                         ? std::min(last_count, max_count) * (1. + kTolerance)
                         : max_count;

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

  auto const foreach_train = [&](service_class const clasz, auto&& fn) {
    for (auto const& e :
         edge_index_[static_cast<service_class_t>(clasz)]->edges(area)) {
      for (auto i = 0U; i < e->m_.route_edge_.conns_.size(); ++i) {
        auto const& c = e->m_.route_edge_.conns_[i];
        if (c.valid_ == 0U || c.a_time_ < start_time || c.d_time_ > end_time) {
          continue;
        }

        auto const k = ev_key{e, i, event_type::DEP};
        float const distance = get_or_create_route_distance(k);
        fn(train{k, distance});
      }
    }
  };

  auto const concat_and_check_limit = [&](auto& trains, auto& other) {
    auto const cleanup = utl::make_finally([&] { other.clear(); });

    if (trains.empty()) {
      std::swap(trains, other);  // "concat"

      if (trains.size() > limit) {
        std::sort(begin(trains), end(trains));
        trains.resize(std::min(trains.size(), static_cast<size_t>(limit)));
        return true;
      }
    } else {
      if (trains.size() + other.size() > limit) {
        return true;
      } else {
        utl::concat(trains, other);
      }
    }
    return false;
  };

  std::shared_lock lock(mutex_);
  std::vector<train> result_trains;
  std::vector<train> clasz_trains;
  for (auto clasz = service_class::AIR; clasz < service_class::NUM_CLASSES;
       ++clasz) {
    if (!path::should_display(clasz, zoom_level,
                              std::numeric_limits<float>::infinity())) {
      continue;
    }

    foreach_train(clasz, [&](auto const& t) {
      if (path::should_display(clasz, zoom_level, t.route_distance_)) {
        clasz_trains.push_back(t);
      }
    });

    if (concat_and_check_limit(result_trains, clasz_trains)) {
      return result_trains;
    }
  }

  return result_trains;
}

}  // namespace motis::railviz
