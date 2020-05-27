#pragma once

#include <memory>

#include "motis/path/prepare/schedule/stations.h"
#include "motis/path/prepare/strategy/routing_strategy.h"

namespace motis::path {

struct path_routing {
  path_routing();
  ~path_routing();

  path_routing(path_routing const&) noexcept = delete;
  path_routing& operator=(path_routing const&) noexcept = delete;
  path_routing(path_routing&&) noexcept = default;
  path_routing& operator=(path_routing&&) noexcept = default;

  std::vector<routing_strategy*> const& strategies_for(
      source_spec::category) const;
  routing_strategy* get_stub_strategy() const;

  struct strategies;
  std::unique_ptr<strategies> strategies_;
};

path_routing make_path_routing(station_index const&,
                               std::string const& osm_path,
                               std::string const& osrm_path);

}  // namespace motis::path
