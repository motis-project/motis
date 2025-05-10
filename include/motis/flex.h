#pragma once

#include "osr/location.h"
#include "osr/types.h"

#include "nigiri/routing/query.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis {

namespace ep {
struct routing;
}

void add_flex_td_offsets(ep::routing const&,
                         osr::location const&,
                         osr::direction,
                         double const max_matching_distance,
                         std::chrono::seconds const max,
                         nigiri::routing::start_time_t const&,
                         nigiri::routing::td_offsets_t& ret);

}  // namespace motis
