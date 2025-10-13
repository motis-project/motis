#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis::odm {

constexpr auto const kODMTransferBuffer = nigiri::duration_t{5};
constexpr auto const kWalkTransportModeId =
    static_cast<nigiri::transport_mode_id_t>(api::ModeEnum::WALK);

bool by_stop(nigiri::routing::start const&, nigiri::routing::start const&);

enum which_mile { kFirstMile, kLastMile };

bool is_odm_leg(nigiri::routing::journey::leg const&,
                nigiri::transport_mode_id_t);

bool uses_odm(nigiri::routing::journey const&, nigiri::transport_mode_id_t);

bool is_pure_pt(nigiri::routing::journey const&);

bool is_direct_odm(nigiri::routing::journey const&);

nigiri::duration_t odm_time(nigiri::routing::journey::leg const&);

nigiri::duration_t odm_time(nigiri::routing::journey const&);

nigiri::duration_t pt_time(nigiri::routing::journey const&);

nigiri::duration_t duration(nigiri::routing::start const&);

std::string odm_label(nigiri::routing::journey const&);

}  // namespace motis::odm