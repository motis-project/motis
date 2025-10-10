#pragma once

#include "osr/routing/profile.h"

#include "nigiri/types.h"

namespace motis {

constexpr auto const kOdmTransportModeId =
    static_cast<nigiri::transport_mode_id_t>(osr::kNumProfiles);
constexpr auto const kRideSharingTransportModeId =
    static_cast<nigiri::transport_mode_id_t>(osr::kNumProfiles + 1U);
constexpr auto const kGbfsTransportModeIdOffset =
    static_cast<nigiri::transport_mode_id_t>(osr::kNumProfiles + 2U);
constexpr auto const kFlexModeIdOffset =
    static_cast<nigiri::transport_mode_id_t>(1'000'000U);

}  // namespace motis