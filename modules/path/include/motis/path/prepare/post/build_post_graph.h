#pragma once

#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

post_graph build_post_graph(mcd::unique_ptr<mcd::vector<station_seq>>);

}  // namespace motis::path
