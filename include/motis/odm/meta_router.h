#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "osr/location.h"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

#include "motis/place.h"

namespace nigiri::routing {
struct query;
struct journey;
}  // namespace nigiri::routing

namespace motis::ep {
struct routing;
}

namespace motis::odm {

std::optional<std::vector<nigiri::routing::journey>> odm_routing(
    ep::routing const& r,
    api::plan_params const& query,
    std::vector<api::ModeEnum> const& pre_transit_modes,
    std::vector<api::ModeEnum> const& post_transit_modes,
    std::vector<api::ModeEnum> const& direct_modes,
    std::variant<osr::location, tt_location> const& from,
    std::variant<osr::location, tt_location> const& to,
    api::Place const& from_p,
    api::Place const& to_p,
    nigiri::routing::query const& start_time);

}  // namespace motis::odm