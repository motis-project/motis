#pragma once

#include "motis-api/motis-api.h"

#include "nigiri/routing/clasz_mask.h"

namespace motis {

nigiri::routing::clasz_mask_t to_clasz_mask(std::vector<api::ModeEnum> const&);

}