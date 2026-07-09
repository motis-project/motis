#pragma once
#include <functional>
#include <vector>

#include "itinerary_id.h"

namespace motis::qa {

constexpr auto kMaxRating = std::numeric_limits<double>::max();
constexpr auto kMinRating = std::numeric_limits<double>::min();

double rate(std::vector<api::Itinerary> const&,
            std::vector<api::Itinerary> const&,
            std::vector<std::function<double(api::Itinerary const&)>> const&);

}  // namespace motis::qa