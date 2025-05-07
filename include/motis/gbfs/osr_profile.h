#pragma once

#include "osr/routing/profile.h"

#include "motis/gbfs/data.h"

namespace motis::gbfs {

osr::search_profile get_osr_profile(provider_products const&);

}  // namespace motis::gbfs
