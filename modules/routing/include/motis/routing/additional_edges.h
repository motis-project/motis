#pragma once

#include <vector>

#include "motis/protocol/RoutingRequest_generated.h"

namespace motis {

struct edge;
struct schedule;

namespace routing {

std::vector<edge> create_additional_edges(
    flatbuffers::Vector<flatbuffers::Offset<AdditionalEdgeWrapper>> const*,
    schedule const&);

}  // namespace routing
}  // namespace motis
