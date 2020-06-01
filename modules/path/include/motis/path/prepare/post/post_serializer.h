#pragma once

#include <limits>
#include <vector>

namespace motis::path {

struct db_builder;
struct post_graph;

void serialize_post_graph(post_graph&, db_builder&);

struct atomic_path;
struct post_segment_id;

std::vector<std::pair<atomic_path*, bool>> reconstruct_path(
    post_segment_id const&,
    size_t sanity_check_limit = std::numeric_limits<size_t>::max());

}  // namespace motis::path
