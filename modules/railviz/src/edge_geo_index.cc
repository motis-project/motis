#include "motis/railviz/edge_geo_index.h"

#include <algorithm>
#include <set>
#include <vector>

#include "boost/geometry/geometries/box.hpp"
#include "boost/geometry/geometries/point.hpp"
#include "boost/geometry/geometries/segment.hpp"
#include "boost/geometry/index/rtree.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using coordinate_system = bg::cs::spherical_equatorial<bg::degree>;
using spherical_point = bg::model::point<double, 2, coordinate_system>;
using box = bg::model::box<spherical_point>;
using value = std::pair<box, int>;
using rtree = bgi::rtree<value, bgi::quadratic<16>>;

namespace motis::railviz {

spherical_point to_point(geo::coord const c) {
  return spherical_point{c.lng_, c.lat_};
}

geo::coord to_coord(spherical_point const p) {
  return {p.get<0>(), p.get<1>()};
}

box bounding_box(spherical_point const c1, spherical_point const c2) {
  bg::model::segment<spherical_point> s(c1, c2);
  box b{};
  bg::envelope(s, b);
  return b;
}

class edge_geo_index::impl {
public:
  impl(int clasz, schedule const& s,
       mcd::hash_map<std::pair<int, int>, geo::box> const& bounding_boxes)
      : no_match_(0),
        clasz_(clasz),
        sched_(s),
        bounding_boxes_(bounding_boxes) {
    init_rtree();

    if (no_match_ != 0) {
      LOG(logging::warn) << "clasz " << clasz
                         << ": station pairs no bounding box: " << no_match_
                         << "/" << station_pairs_.size();
    }
  }

  std::vector<edge const*> edges(geo::box const& b) const {
    std::vector<value> result_n;
    auto const bounds = bounding_box(to_point(b.first), to_point(b.second));
    rtree_.query(bgi::intersects(bounds), std::back_inserter(result_n));

    std::vector<edge const*> edges;
    for (auto const& station_pair_idx : result_n) {
      retrieve_edges(station_pairs_[station_pair_idx.second], edges);
    }
    return edges;
  }

  geo::box get_bounds() const {
    auto const bounds = rtree_.bounds();
    return {to_coord(bounds.min_corner()), to_coord(bounds.max_corner())};
  }

private:
  static std::pair<int, int> key(edge const& e) {
    auto const from = e.from_->get_station()->id_;
    auto const to = e.to_->get_station()->id_;
    return {std::min(from, to), std::max(from, to)};
  }

  void init_rtree() {
    std::vector<value> entries;
    std::set<std::pair<int, int>> indexed_station_pairs;
    for (const auto& node : sched_.station_nodes_) {
      add_edges_of_station(node.get(), indexed_station_pairs, entries);
    }
    rtree_ = rtree(entries);
  }

  void add_edges_of_station(
      station_node const* sn,
      std::set<std::pair<int, int>>& indexed_station_pairs,
      std::vector<value>& entries) {
    assert(sn->is_station_node());
    sn->for_each_route_node([&](node const* route_node) {
      add_edges_of_route_node(route_node, indexed_station_pairs, entries);
    });
  }

  void add_edges_of_route_node(
      node const* route_node,
      std::set<std::pair<int, int>>& indexed_station_pairs,
      std::vector<value>& entries) {
    assert(route_node->is_route_node());
    for (auto const& e : route_node->edges_) {
      if (!is_relevant(e)) {
        continue;
      }

      auto const station_pair = key(e);
      if (!indexed_station_pairs.insert(station_pair).second) {
        continue;
      }

      entries.emplace_back(get_bounding_box(station_pair),
                           static_cast<int>(station_pairs_.size()));
      station_pairs_.emplace_back(station_pair);
    }
  }

  void retrieve_edges(std::pair<int, int> const& stations,
                      std::vector<edge const*>& edges) const {
    auto const dir1 = std::make_pair(stations.second, stations.first);
    auto const dir2 = std::make_pair(stations.first, stations.second);

    for (auto const& d : {dir1, dir2}) {
      auto const from = sched_.station_nodes_[d.first].get();
      auto const to = sched_.station_nodes_[d.second].get();

      from->for_each_route_node([&](node const* route_node) {
        for (auto const& e : route_node->edges_) {
          if (!is_relevant(e) || e.to_->get_station() != to) {
            continue;
          }

          edges.emplace_back(&e);
        }
      });
    }
  }

  spherical_point station_coords(unsigned station_idx) {
    auto const& station = *sched_.stations_[station_idx];
    return spherical_point(station.length_, station.width_);
  }

  box get_bounding_box(std::pair<int, int> const& stations) {
    auto const it = bounding_boxes_.find(stations);
    no_match_ += (it == end(bounding_boxes_) ? 1 : 0);
    return it == end(bounding_boxes_)
               ? bounding_box(station_coords(stations.first),
                              station_coords(stations.second))
               : bounding_box(to_point(it->second.first),
                              to_point(it->second.second));
  }

  inline bool is_relevant(edge const& e) const {
    return !e.empty() && e.m_.route_edge_.conns_[0].full_con_->clasz_ == clasz_;
  }

  int no_match_;
  int clasz_;
  schedule const& sched_;
  std::vector<std::pair<int, int>> station_pairs_;
  mcd::hash_map<std::pair<int, int>, geo::box> const& bounding_boxes_;
  rtree rtree_;
};

edge_geo_index::edge_geo_index(
    int clasz, schedule const& s,
    mcd::hash_map<std::pair<int, int>, geo::box> const& bounding_boxes)
    : impl_(new impl(clasz, s, bounding_boxes)) {}

edge_geo_index::~edge_geo_index() = default;

std::vector<edge const*> edge_geo_index::edges(geo::box const& area) const {
  return impl_->edges(area);
}

geo::box edge_geo_index::get_bounds() const { return impl_->get_bounds(); }

}  // namespace motis::railviz
