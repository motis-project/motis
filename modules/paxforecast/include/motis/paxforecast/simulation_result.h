#pragma once

#include <numeric>
#include <set>
#include <vector>

#include "motis/hash_map.h"

#include "motis/paxmon/graph.h"

namespace motis::paxforecast {

struct simulation_result {

  bool is_over_capacity() const { return !edges_over_capacity_.empty(); }

  std::size_t edge_count_over_capacity() const {
    return edges_over_capacity_.size();
  }

  std::size_t total_passengers_over_capacity() const {
    return std::accumulate(begin(edges_over_capacity_),
                           end(edges_over_capacity_), 0ULL,
                           [](auto const sum, auto const e) {
                             return sum + e->passengers_over_capacity();
                           });
  }

  std::set<trip const*> trips_over_capacity() const {
    std::set<trip const*> trips;
    for (auto const e : edges_over_capacity_) {
      trips.insert(e->get_trip());
    }
    return trips;
  }

  mcd::hash_map<trip const*, std::vector<motis::paxmon::edge*>>
  trips_over_capacity_with_edges() const {
    mcd::hash_map<trip const*, std::vector<motis::paxmon::edge*>> edges_by_trip;
    for (auto const e : edges_over_capacity_) {
      edges_by_trip[e->get_trip()].push_back(e);
    }
    return edges_by_trip;
  }

  mcd::hash_map<motis::paxmon::edge*, std::uint16_t> additional_passengers_;
  std::set<motis::paxmon::edge*> edges_over_capacity_;
};

}  // namespace motis::paxforecast
