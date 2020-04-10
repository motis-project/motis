#pragma once

#include "motis/path/prepare/osm/osm_graph.h"

namespace motis::path {

std::vector<std::vector<osm_edge const*>> shortest_paths(
    osm_graph const&, size_t const& from, std::vector<size_t> const& to,
    bool ignore_limit = false);

std::vector<double> shortest_path_distances(osm_graph const& graph,
                                            size_t const& from,
                                            std::vector<size_t> const& to);

}  // namespace motis::path
