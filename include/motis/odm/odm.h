#pragma once

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis::odm {

constexpr auto const kODM =
    static_cast<nigiri::transport_mode_id_t>(api::ModeEnum::ODM);
constexpr auto const kWalk =
    static_cast<nigiri::transport_mode_id_t>(api::ModeEnum::WALK);

}  // namespace motis::odm