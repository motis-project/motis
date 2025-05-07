#pragma once

#include <string_view>
#include <variant>

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

#include "osr/location.h"
#include "osr/routing/route.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/place.h"
#include "motis/street_routing.h"
#include "motis/types.h"

namespace motis {

double get_level(osr::ways const*,
                 osr::platforms const*,
                 platform_matches_t const*,
                 nigiri::location_idx_t);

api::Itinerary journey_to_response(osr::ways const*,
                                   osr::lookup const*,
                                   osr::platforms const*,
                                   nigiri::timetable const&,
                                   tag_lookup const&,
                                   elevators const* e,
                                   nigiri::rt_timetable const*,
                                   platform_matches_t const* matches,
                                   osr::elevation_storage const*,
                                   nigiri::shapes_storage const*,
                                   gbfs::gbfs_routing_data&,
                                   api::PedestrianProfileEnum,
                                   api::ElevationCostsEnum,
                                   nigiri::routing::journey const&,
                                   place_t const& start,
                                   place_t const& dest,
                                   street_routing_cache_t&,
                                   osr::bitvec<osr::node_idx_t>* blocked_mem,
                                   bool detailed_transfers,
                                   bool with_fares,
                                   double timetable_max_matching_distance,
                                   double max_matching_distance,
                                   unsigned api_version);

}  // namespace motis
