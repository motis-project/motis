#include "motis/odm/mix.h"

namespace motis::odm {

std::int32_t tally(std::int32_t const x,
                   std::vector<cost_threshold> const& ct) {
  auto acc = std::int32_t{0};
  for (auto i = 0U; i < ct.size() && ct[i].threshold_ < x; ++i) {
    auto const valid_until = i + 1U == ct.size()
                                 ? std::numeric_limits<std::int32_t>::max()
                                 : ct[i + 1U].threshold_;
    acc += (std::min(x, valid_until) - ct[i].threshold_) * ct[i].cost_;
  }
  return acc;
}

void cost_domination(n::pareto_set<n::routing::journey> const& base_journeys,
                     std::vector<n::routing::journey>& odm_journeys) {
  auto const start_cost = [](auto const& j) {

  };
}

void productivity_domination(std::vector<n::routing::journey>& odm_journeys) {}

void mix(n::pareto_set<n::routing::journey> const& base_journeys,
         std::vector<n::routing::journey>& odm_journeys) {
  cost_domination(base_journeys, odm_journeys);
  productivity_domination(odm_journeys);
  odm_journeys.append_range(base_journeys);
}

}  // namespace motis::odm