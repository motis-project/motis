#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/csa/csa_implementation_type.h"
#include "motis/csa/csa_journey.h"
#include "motis/csa/csa_query.h"
#include "motis/csa/csa_timetable.h"
#include "motis/csa/response.h"

#include "motis/protocol/RoutingRequest_generated.h"

namespace motis::csa {

response run_csa_search(schedule const&, csa_timetable const&, csa_query const&,
                        motis::routing::SearchType, implementation_type,
                        bool use_profile_search);

}  // namespace motis::csa
