#pragma once

#include <functional>
#include <ostream>

#include "motis/core/journey/journey.h"

namespace motis {

bool check_journey(journey const& j,
                   std::function<std::ostream&(bool)> const& report_error);

}  // namespace motis
