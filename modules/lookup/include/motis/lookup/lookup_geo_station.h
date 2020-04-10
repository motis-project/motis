#pragma once

#include <vector>

#include "geo/point_rtree.h"

#include "motis/core/schedule/schedule.h"

#include "motis/protocol/Message_generated.h"

namespace motis::lookup {

flatbuffers::Offset<LookupGeoStationResponse> lookup_geo_stations_id(
    flatbuffers::FlatBufferBuilder&, geo::point_rtree const& station_geo_index,
    schedule const&, LookupGeoStationIdRequest const*);

flatbuffers::Offset<LookupGeoStationResponse> lookup_geo_stations(
    flatbuffers::FlatBufferBuilder&, geo::point_rtree const& station_geo_index,
    schedule const&, LookupGeoStationRequest const*);

}  // namespace motis::lookup
