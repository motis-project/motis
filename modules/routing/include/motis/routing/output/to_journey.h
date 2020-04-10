#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"
#include "motis/routing/output/stop.h"
#include "motis/routing/output/transport.h"

namespace motis::routing::output {

journey::transport generate_journey_transport(unsigned int from,
                                              unsigned int to,
                                              intermediate::transport const& t,
                                              schedule const& sched,
                                              unsigned int route_id);

std::vector<journey::transport> generate_journey_transports(
    std::vector<intermediate::transport> const&, schedule const&);

std::vector<journey::trip> generate_journey_trips(
    std::vector<intermediate::transport> const&, schedule const&);

std::vector<journey::stop> generate_journey_stops(
    std::vector<intermediate::stop> const& stops, schedule const& sched);

std::vector<journey::ranged_attribute> generate_journey_attributes(
    std::vector<intermediate::transport> const& transports);

}  // namespace motis::routing::output
