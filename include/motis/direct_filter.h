#pragma once

#include <vector>

#include "nigiri/routing/journey.h"

#include "motis-api/motis-api.h"

namespace motis {

void direct_filter(std::vector<api::Itinerary> const& direct,
                   std::vector<nigiri::routing::journey>&);

}  // namespace motis