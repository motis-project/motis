#pragma once

#include "motis/transfers/platform/platform.h"

#include "ppr/routing/input_location.h"

namespace motis::transfers {

// Returns the equivalent OSM_TYPE of PPR.
// Default: ::ppr::routing::osm_namespace::NODE
::ppr::routing::osm_namespace to_ppr_osm_type(osm_type const&);

// Builds an `input_location` for the ppr::routing_graph that represents the
// given platform.
::ppr::routing::input_location to_input_location(platform const&);

}  // namespace motis::transfers
