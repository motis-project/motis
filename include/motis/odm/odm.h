#pragma once

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis::odm {

static constexpr auto const kODMTransferBuffer = nigiri::duration_t{5};
constexpr auto const kODM =
    static_cast<nigiri::transport_mode_id_t>(api::ModeEnum::ODM);
constexpr auto const kWalk =
    static_cast<nigiri::transport_mode_id_t>(api::ModeEnum::WALK);

bool is_odm_leg(nigiri::routing::journey::leg const&);

}  // namespace motis::odm