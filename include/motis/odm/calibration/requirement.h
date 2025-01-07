#pragma once

#include <bitset>
#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

namespace motis::odm {

struct requirement {
  std::vector<nigiri::routing::journey> get_expected() const;

  nigiri::pareto_set<nigiri::routing::journey> pt_{};
  std::vector<nigiri::routing::journey> odm_{};
  std::bitset<64> odm_to_dom_{};
};

}  // namespace motis::odm