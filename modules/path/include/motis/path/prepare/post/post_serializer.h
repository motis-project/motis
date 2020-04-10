#pragma once

namespace motis::path {

struct db_builder;
struct post_graph;

void serialize_post_graph(post_graph&, db_builder&);

}  // namespace motis::path
