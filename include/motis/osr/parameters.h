#pragma once

#include <optional>
#include <string>
#include <vector>

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

// Shared-mobility (gbfs/rental) filter options for one access/egress direction,
// mirroring the plan endpoint's pre-/post-transit rental parameters.
struct rental_options {
  std::optional<std::vector<api::RentalFormFactorEnum>> form_factors_{};
  std::optional<std::vector<api::RentalPropulsionTypeEnum>> propulsion_types_{};
  std::optional<std::vector<std::string>> providers_{};
  std::optional<std::vector<std::string>> provider_groups_{};
  bool ignore_return_constraints_{false};
};

struct osr_parameters {
  constexpr static auto const kFootSpeed = 1.2F;
  constexpr static auto const kWheelchairSpeed = 0.8F;
  constexpr static auto const kBikeSpeed = 4.2F;
  constexpr static auto const kCarSpeed = 28.0F;
  constexpr static auto const kBusSpeed = 28.0F;
  constexpr static auto const kRailwaySpeed = 28.0F;
  constexpr static auto const kFerrySpeed = 28.0F;

  float const pedestrian_speed_{kFootSpeed};
  float const cycling_speed_{kBikeSpeed};
  bool const use_wheelchair_{false};
};

osr_parameters get_osr_parameters(api::plan_params const&);

osr_parameters get_osr_parameters(api::refreshItinerary_params const&);

osr_parameters get_osr_parameters(api::oneToAll_params const&);

osr_parameters get_osr_parameters(api::oneToMany_params const&);

osr_parameters get_osr_parameters(api::OneToManyParams const&);

osr_parameters get_osr_parameters(api::oneToManyIntermodal_params const&);

osr_parameters get_osr_parameters(api::OneToManyIntermodalParams const&);

osr::profile_parameters to_profile_parameters(osr::search_profile,
                                              osr_parameters const&);
}  // namespace motis
