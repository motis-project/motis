#pragma once

#include <chrono>

#include "osr/routing/profile.h"

#include "motis/osr/parameters.h"

namespace motis {

double get_max_distance(osr::search_profile,
                        osr_parameters const&,
                        std::chrono::seconds);

// Upper bound on the distance a search with the given cost budget can cover,
// derived from the lower bound osr uses for A*. Unlike get_max_distance, this
// accounts for ways the profile traverses faster than its nominal speed.
double get_distance_upper_bound(osr::search_profile,
                                osr_parameters const&,
                                std::chrono::seconds);

}  // namespace motis
