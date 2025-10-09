#pragma once

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis {

api::ModeEnum to_mode(nigiri::clasz, unsigned api_version);

std::vector<api::ModeEnum> to_modes(nigiri::routing::clasz_mask_t,
                                    unsigned api_version);

}  // namespace motis