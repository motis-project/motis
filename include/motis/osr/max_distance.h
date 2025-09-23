#pragma once

#include <chrono>

#include "osr/routing/profile.h"

namespace motis {

double get_max_distance(osr::search_profile, std::chrono::seconds);

}  // namespace motis