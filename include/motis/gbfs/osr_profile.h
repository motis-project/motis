#pragma once

#include "osr/routing/profile.h"

#include "motis/gbfs/data.h"

namespace motis::gbfs {

osr::search_profile get_osr_profile(vehicle_form_factor const&);

}  // namespace motis::gbfs
