#pragma once

#include <memory>

#include "motis/path/prepare/osm/osm_data.h"
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

path_routing make_path_routing(mcd::vector<station_seq> const&,
                               station_index const&, osm_data const&,
                               std::string const& osrm_path);

}  // namespace motis::path
