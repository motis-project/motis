#pragma once

#include <algorithm>

#include "utl/verify.h"

#include "motis/core/journey/journey.h"

#include "motis/revise/extern_interchange.h"

namespace motis::revise {

std::vector<extern_interchange> get_interchanges(journey const& j);

}  // namespace motis::revise
