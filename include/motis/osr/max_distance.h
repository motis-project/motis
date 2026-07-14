#pragma once

#include <chrono>

#include "osr/routing/profile.h"

#include "motis/osr/parameters.h"

namespace motis {

double get_max_distance(osr::search_profile,
                        osr_parameters const&,
                        std::chrono::seconds);

}  // namespace motis
