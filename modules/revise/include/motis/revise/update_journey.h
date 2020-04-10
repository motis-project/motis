#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/revise/extern_interchange.h"

namespace motis::revise {

journey update_journey(schedule const&, journey const&);

std::vector<extern_interchange> get_interchanges(journey const&);

}  // namespace motis::revise
