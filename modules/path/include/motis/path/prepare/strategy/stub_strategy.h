#pragma once

#include <vector>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/path/prepare/strategy/routing_strategy.h"
#include "motis/path/prepare/tuning_parameters.h"

namespace motis::path {

struct stub_strategy : public routing_strategy {
  stub_strategy(strategy_id_t strategy_id, source_spec spec,
                std::vector<station> const& stations)
      : routing_strategy(strategy_id, spec) {
    auto id = 0U;
    for (auto const& station : stations) {
      stations_to_refs_[station.id_] = {{strategy_id, id++, station.pos_, 0}};
    }
  }

  ~stub_strategy() override = default;

  stub_strategy(stub_strategy const&) noexcept = delete;
  stub_strategy& operator=(stub_strategy const&) noexcept = delete;
  stub_strategy(stub_strategy&&) noexcept = delete;
  stub_strategy& operator=(stub_strategy&&) noexcept = delete;

  std::vector<node_ref> const& close_nodes(
      std::string const& station_id) const override {
    auto const it = stations_to_refs_.find(station_id);
    utl::verify(it != end(stations_to_refs_), "stub: unknown station!");
    return it->second;
  }

  routing_result_matrix find_routes(
      std::vector<node_ref> const& from,
      std::vector<node_ref> const& to) const override {
    routing_result_matrix mat{from.size(), to.size()};
    mat.foreach ([&](auto const from_idx, auto const to_idx, auto& weight) {
      weight = geo::distance(from[from_idx].coords_, to[to_idx].coords_) *
               kStubStrategyPenaltyFactor;
    });
    return mat;
  }

  osm_path get_path(node_ref const& from, node_ref const& to) const override {
    return osm_path{mcd::vector<geo::latlng>{from.coords_, to.coords_}};
  }

  mcd::hash_map<std::string, std::vector<node_ref>> stations_to_refs_;
};

}  // namespace motis::path
