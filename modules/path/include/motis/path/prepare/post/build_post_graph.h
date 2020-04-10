#pragma once

#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/resolve/resolved_station_seq.h"

namespace motis::path {

post_graph build_post_graph(std::vector<resolved_station_seq>);

}  // namespace motis::path
