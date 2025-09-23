#pragma once

#include "osr/routing/mode.h"
#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

api::ModeEnum to_mode(osr::mode);
osr::search_profile to_profile(api::ModeEnum,
                               api::PedestrianProfileEnum,
                               api::ElevationCostsEnum);

}  // namespace motis