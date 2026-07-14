#pragma once

#include <chrono>
#include <string_view>
#include <variant>
#include <vector>

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
#include "motis/osr/parameters.h"
#include "motis/osr/street_routing.h"
#include "motis/place.h"
#include "motis/types.h"

namespace motis {

api::ModeEnum to_mode(nigiri::transport_mode_id_t);

api::ModeEnum to_mode(osr::search_profile);

double get_level(osr::ways const*,
                 osr::platforms const*,
                 platform_matches_t const*,
                 nigiri::location_idx_t);

std::optional<std::vector<api::Alert>> get_alerts(
    nigiri::rt::frun const&,
    std::optional<std::pair<nigiri::rt::run_stop, nigiri::event_type>> const&,
    bool fuzzy_stop,
    std::optional<std::vector<std::string>> const& language);

struct query_alternatives {
  nigiri::routing::query const& query;
  std::size_t num_alternatives;
};
using alternatives_context = std::variant<
    // no alternatives
    std::monostate,
    // compute alternatives from the query+journey context
    query_alternatives,
    // precomputed
    std::vector<nigiri::routing::journey>>;

api::Itinerary journey_to_response(
    osr::ways const*,
    osr::lookup const*,
    osr::platforms const*,
    nigiri::timetable const&,
    tag_lookup const&,
    flex::flex_areas const*,
    elevators const* e,
    nigiri::rt_timetable const*,
    platform_matches_t const* matches,
    osr::elevation_storage const*,
    nigiri::shapes_storage const*,
    gbfs::gbfs_routing_data&,
    adr_ext const*,
    tz_map_t const*,
    nigiri::routing::journey const&,
    place_t const& start,
    place_t const& dest,
    street_routing_cache_t&,
    osr::bitvec<osr::node_idx_t>* blocked_mem,
    bool car_transfers,
    osr_parameters const&,
    api::PedestrianProfileEnum,
    api::ElevationCostsEnum,
    bool join_interlined_legs,
    bool detailed_transfers,
    bool detailed_legs,
    bool with_fares,
    bool with_scheduled_skipped_stops,
    double timetable_max_matching_distance,
    double max_matching_distance,
    unsigned api_version,
    bool ignore_start_rental_return_constraints,
    bool ignore_dest_rental_return_constraints,
    std::optional<std::vector<std::string>> const& language,
    bool const set_itinerary_id_field = true,
    alternatives_context const& alternatives = {},
    std::chrono::nanoseconds* fares_time = nullptr);

}  // namespace motis
