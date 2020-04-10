#pragma once

#include <memory>
#include <vector>

#include "motis/hash_map.h"

#include "motis/railviz/geo.h"

namespace motis {
class edge;
struct schedule;
}  // namespace motis

namespace motis::railviz {

struct edge_geo_index {
public:
  edge_geo_index(int clasz, schedule const&,
                 mcd::hash_map<std::pair<int, int>, geo::box> const&);

  edge_geo_index(edge_geo_index const&) = delete;
  edge_geo_index& operator=(edge_geo_index const&) = delete;

  edge_geo_index(edge_geo_index&&) = delete;
  edge_geo_index& operator=(edge_geo_index&&) = delete;

  virtual ~edge_geo_index();

  std::vector<edge const*> edges(geo::box const& area) const;
  geo::box get_bounds() const;

private:
  class impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::railviz
