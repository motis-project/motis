#pragma once

#include "osr/location.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis {

namespace ep {
struct routing;
}

void add_flex_td_offsets(ep::routing const&,
                         osr::location const&,
                         osr::direction,
                         api::PedestrianProfileEnum,
                         double const max_matching_distance,
                         std::chrono::seconds const max);

}  // namespace motis
