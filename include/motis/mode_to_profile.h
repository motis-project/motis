#pragma once

#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

osr::search_profile to_profile(api::ModeEnum, bool wheelchair);

}  // namespace motis