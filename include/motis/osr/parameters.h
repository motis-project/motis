#pragma once

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

struct osr_parameters {
  constexpr static auto const kFootSpeed = 1.2F;
  constexpr static auto const kWheelchairSpeed = 0.8F;
  constexpr static auto const kBikeSpeed = 4.2F;

  float const pedestrian_speed_{kFootSpeed};
  float const cycling_speed_{kBikeSpeed};
  bool const use_wheelchair_{false};
};

osr_parameters get_osr_parameters(api::plan_params const&);

osr_parameters get_osr_parameters(api::oneToAll_params const&);

osr_parameters get_osr_parameters(api::oneToMany_params const&);

osr_parameters get_osr_parameters(api::OneToManyParams const&);

osr::profile_parameters to_profile_parameters(osr::search_profile,
                                              osr_parameters const&);
}  // namespace motis
