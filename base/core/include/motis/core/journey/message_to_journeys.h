#pragma once

#include <vector>

namespace motis {

struct journey;
struct Connection;  // NOLINT

namespace routing {
struct RoutingResponse;  // NOLINT
}  // namespace routing

std::vector<journey> message_to_journeys(routing::RoutingResponse const*);
journey convert(Connection const*);

}  // namespace motis
