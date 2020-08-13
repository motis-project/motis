#pragma once

#include "motis/raptor/raptor_query.h"

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

namespace motis::raptor {

using namespace motis::routing;

base_query get_base_query(RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_schedule const& raptor_sched);

}  // namespace motis::raptor