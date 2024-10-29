#pragma once

#include "osr/location.h"

#include "motis-api/motis-api.h"

#include "motis/fwd.h"
#include "motis/gbfs/data.h"
#include "motis/journey_to_response.h"
#include "motis/place.h"

namespace motis::gbfs {

api::Itinerary route(osr::ways const& w,
                     osr::lookup const& l,
                     gbfs::gbfs_data const& gbfs,
                     place_t const& from,
                     place_t const& to,
                     gbfs::gbfs_provider_idx_t);

}  // namespace motis::gbfs