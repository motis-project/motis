#pragma once

#include <memory>

#include "motis/path/prepare/schedule/stations.h"
#include "motis/path/prepare/strategy/routing_strategy.h"

namespace motis::path {

struct osrm_strategy : public routing_strategy {
  osrm_strategy(strategy_id_t, source_spec,  //
                std::vector<station> const&, std::string const& osrm_path);
  ~osrm_strategy() override;

  osrm_strategy(osrm_strategy const&) noexcept = delete;
  osrm_strategy& operator=(osrm_strategy const&) noexcept = delete;
  osrm_strategy(osrm_strategy&&) noexcept = delete;
  osrm_strategy& operator=(osrm_strategy&&) noexcept = delete;

  std::vector<node_ref> const& close_nodes(
      std::string const& station_id) const override;

  routing_result_matrix find_routes(
      std::vector<node_ref> const& from,
      std::vector<node_ref> const& to) const override;

  osm_path get_path(node_ref const& from, node_ref const& to) const override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::path
