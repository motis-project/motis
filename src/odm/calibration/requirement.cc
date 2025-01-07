#include "motis/odm/calibration/requirement.h"

#include "utl/enumerate.h"

namespace motis::odm {

std::vector<nigiri::routing::journey> requirement::get_expected() const {
  auto exp_mix = std::vector<nigiri::routing::journey>{};

  for (auto const [i, j] : utl::enumerate(odm_)) {
    if (!odm_to_dom_.test(i)) {
      exp_mix.emplace_back(j);
    }
  }

  return exp_mix;
}

}  // namespace motis::odm