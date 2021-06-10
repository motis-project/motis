#pragma once

#include <queue>

#include "boost/container/vector.hpp"
#include "utl/erase_if.h"

#include "motis/hash_map.h"

#include "motis/core/common/dial.h"
#include "build_query.h"

namespace motis::isochrone {

    const bool FORWARDING = true;

    using td_graph = mcd::vector<mcd::vector<edge>>;

    inline td_graph build_graph(mcd::vector<station_node_ptr> const &station_nodes) {
        td_graph g(station_nodes.size());

        return g;
    }


// dijkstra
    void search() {

    }

}

