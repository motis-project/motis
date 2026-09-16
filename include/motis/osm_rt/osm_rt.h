#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/asio/io_context.hpp"

#include "osr/types.h"

#include "motis/fwd.h"

namespace motis {

// Snapshot built from the latest update of all configured OSM-RT feeds.
// Swapped atomically by the update loop (like data::rt_).
struct osm_rt_data {
  osr::bitvec<osr::node_idx_t> blocked_;
};

// Parses OSM-RT feed messages and matches the geometry of entities carrying
// access-blocking tags (e.g. motor_vehicle=no) to street routing nodes.
osr::bitvec<osr::node_idx_t> osm_rt_blocked_nodes(
    osr::ways const&,
    osr::lookup const&,
    std::vector<std::string> const& feed_messages);

void run_osm_rt_update(boost::asio::io_context&, config const&, data&);

}  // namespace motis
