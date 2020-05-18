#include "motis/railviz/train_retriever.h"

#include <algorithm>
#include <set>
#include <vector>

#include "boost/geometry/geometries/box.hpp"
#include "boost/geometry/geometries/point.hpp"
#include "boost/geometry/geometries/segment.hpp"
#include "boost/geometry/index/rtree.hpp"

#include "utl/get_or_create.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/get_schedule.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using coordinate_system = bg::cs::spherical_equatorial<bg::degree>;
using spherical_point = bg::model::point<double, 2, coordinate_system>;
using box = bg::model::box<spherical_point>;
using value = std::pair<box, std::pair<int, int>>;
using rtree = bgi::rtree<value, bgi::quadratic<16>>;

namespace motis::railviz {

spherical_point to_point(geo::coord const c) {
  return spherical_point{c.lng_, c.lat_};
}

box bounding_box(spherical_point const c1, spherical_point const c2) {
  box b{};
  bg::envelope(bg::model::segment<spherical_point>{c1, c2}, b);
  return b;
}

bool is_relevant(edge const& e, int clasz) {
  return !e.empty() && e.m_.route_edge_.conns_[0].full_con_->clasz_ == clasz;
}

spherical_point station_coords(schedule const& sched, unsigned station_idx) {
  auto const& station = sched.stations_[station_idx];
  return spherical_point(station->length_, station->width_);
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
    auto const bounds = bounding_box(to_point(b.first), to_point(b.second));
    tree_.query(bgi::intersects(bounds), std::back_inserter(result_n));

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
               ? bounding_box(station_coords(sched, stations.first),
                              station_coords(sched, stations.second))
               : bounding_box(to_point(it->second.first),
                              to_point(it->second.second));
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

constexpr auto const RELEVANT_CLASSES = NUM_CLASSES - 1;

train_retriever::train_retriever(
    schedule const& sched, mcd::hash_map<std::pair<int, int>, geo::box> boxes) {
  edge_index_.resize(RELEVANT_CLASSES);
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_[clasz] = make_edge_rtree(sched, clasz, boxes);
  }
}

train_retriever::~train_retriever() = default;

void train_retriever::update(rt::RtUpdates const* updates) {
  std::unique_lock lock(mutex_);

  auto const& sched = module::get_schedule();

  std::vector<std::vector<value>> new_values;
  new_values.resize(RELEVANT_CLASSES);

  for (auto const* update : *updates->updates()) {
    if (update->content_type() != rt::Content_RtRerouteUpdate) {
      continue;
    }

    auto const reroute_update =
        reinterpret_cast<rt::RtRerouteUpdate const*>(update->content());
    for (auto const& section :
         access::sections(from_fbs(sched, reroute_update->trip()))) {

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
          bounding_box(station_coords(sched, station_pair.first),
                       station_coords(sched, station_pair.second)),
          station_pair);
    }
  }

  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    edge_index_.at(clasz)->tree_.insert(new_values.at(clasz));
  }
}

int cls_to_min_zoom_level(int c) {
  if (c < 3) {
    return 4;
  } else if (c < 6) {
    return 6;
  } else if (c < 8) {
    return 9;
  } else {
    return 10;
  }
}

std::vector<ev_key> train_retriever::trains(time const from, time const to,
                                            unsigned const max_count,
                                            geo::box const& area,
                                            int zoom_level) {
  std::shared_lock lock(mutex_);
  std::vector<ev_key> connections;
  for (auto clasz = 0U; clasz < RELEVANT_CLASSES; ++clasz) {
    if (zoom_level < cls_to_min_zoom_level(clasz)) {
      goto end;
    }

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
