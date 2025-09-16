#pragma once

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

struct profile_parameters {
  constexpr static auto const kFootSpeed = 1.2F;
  constexpr static auto const kWheelchairSpeed = 0.8F;
  constexpr static auto const kBikeSpeed = 4.2F;

  float const pedestrian_speed_{kFootSpeed};
  float const cycling_speed_{kBikeSpeed};
};

profile_parameters get_parameters(api::plan_params const&);

profile_parameters get_parameters(api::oneToAll_params const&);

osr::profile_parameters build_parameters(osr::search_profile,
                                         profile_parameters const&);
}  // namespace motis
