#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

namespace motis::odm {

namespace n = nigiri;

struct cost_threshold {
  std::int32_t threshold_;
  std::int32_t cost_;
};

std::int32_t tally(std::int32_t, std::vector<cost_threshold> const&);

void mix(n::pareto_set<n::routing::journey> const&,
         std::vector<n::routing::journey>&);

}  // namespace motis::odm