#pragma once

#include <chrono>

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"

#include "motis/osr/parameters.h"

namespace motis {

constexpr double get_max_distance(osr::profile_parameters const&, std::chrono::seconds);

double get_max_distance(osr::search_profile, std::chrono::seconds);

double get_max_distance(osr::search_profile,
                        osr_parameters const&,
                        std::chrono::seconds);

}  // namespace motis
