#pragma once

#include <optional>
#include <vector>

#include "motis/gbfs/data.h"

#include "motis-api/motis-api.h"

namespace motis::gbfs {

api::RentalFormFactorEnum to_api_form_factor(vehicle_form_factor);
vehicle_form_factor from_api_form_factor(api::RentalFormFactorEnum);

api::RentalPropulsionTypeEnum to_api_propulsion_type(propulsion_type);
propulsion_type from_api_propulsion_type(api::RentalPropulsionTypeEnum);

bool segment_matches(
    provider_segment const& seg,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types);

}  // namespace motis::gbfs
