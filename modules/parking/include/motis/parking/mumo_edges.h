#pragma once

#include "flatbuffers/flatbuffers.h"

#include "geo/latlng.h"

#include "ppr/common/location.h"

#include "motis/module/message.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

motis::module::msg_ptr make_geo_station_request(geo::latlng const& pos,
                                                double radius);

motis::module::msg_ptr make_osrm_request(
    geo::latlng const& pos, std::vector<parking_lot> const& destinations,
    std::string const& profile, motis::osrm::Direction direction);

motis::module::msg_ptr make_ppr_request(
    geo::latlng const& pos, std::vector<Position> const& destinations,
    motis::ppr::SearchOptions const* search_options,
    motis::ppr::SearchDirection dir, bool include_steps = false,
    bool include_edges = false, bool include_path = false);

motis::module::msg_ptr make_ppr_request(
    ::ppr::location const& start,
    std::vector<::ppr::location> const& destinations,
    std::string const& profile_name, double const duration_limit,
    motis::ppr::SearchDirection dir, bool include_steps = false,
    bool include_edges = false, bool include_path = false);

motis::module::msg_ptr make_ppr_request(
    geo::latlng const& pos,
    flatbuffers::Vector<flatbuffers::Offset<Station>> const* stations,
    motis::ppr::SearchOptions const* search_options,
    motis::ppr::SearchDirection dir, bool include_steps = false,
    bool include_edges = false, bool include_path = false);

}  // namespace motis::parking
