#pragma once

#include "flatbuffers/flatbuffers.h"

#include "motis/module/message.h"
#include "motis/parking/parking_lot.h"
#include "geo/latlng.h"

namespace motis::parking {

motis::module::msg_ptr make_geo_station_request(geo::latlng const& pos,
                                                double radius);

motis::module::msg_ptr make_osrm_request(
    geo::latlng const& pos, std::vector<parking_lot> const& destinations,
    std::string const& profile, SearchDir);

motis::module::msg_ptr make_ppr_request(
    geo::latlng const& pos, std::vector<Position> const& destinations,
    motis::ppr::SearchOptions const* search_options, SearchDir,
    bool include_steps = false, bool include_edges = false,
    bool include_path = false);

motis::module::msg_ptr make_ppr_request(
    geo::latlng const& pos,
    flatbuffers::Vector<flatbuffers::Offset<Station>> const* stations,
    motis::ppr::SearchOptions const* search_options, SearchDir,
    bool include_steps = false, bool include_edges = false,
    bool include_path = false);

}  // namespace motis::parking
