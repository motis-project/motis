#pragma once

#include "nigiri/types.h"

#include "motis/gbfs/data.h"
#include "motis/gbfs/routing_data.h"

#include "motis-api/motis-api.h"

namespace motis::gbfs {

api::ModeEnum get_gbfs_mode(vehicle_form_factor);
api::ModeEnum get_gbfs_mode(gbfs_data const&, gbfs_segment_ref);
api::ModeEnum get_gbfs_mode(gbfs_routing_data const&,
                            nigiri::transport_mode_id_t);

bool form_factor_matches(api::ModeEnum, gbfs::vehicle_form_factor);

}  // namespace motis::gbfs
