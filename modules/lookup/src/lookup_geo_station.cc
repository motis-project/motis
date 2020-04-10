#include "motis/lookup/lookup_geo_station.h"

#include "motis/core/access/station_access.h"
#include "motis/lookup/util.h"

using namespace flatbuffers;

namespace motis::lookup {

Offset<LookupGeoStationResponse> lookup_geo_stations_id(
    FlatBufferBuilder& b, geo::point_rtree const& station_geo_index,
    schedule const& sched, LookupGeoStationIdRequest const* req) {
  auto const& station = get_station(sched, req->station_id()->str());
  return CreateLookupGeoStationResponse(
      b, b.CreateVector(utl::to_vec(
             station_geo_index.in_radius({station->lat(), station->lng()},
                                         req->min_radius(), req->max_radius()),
             [&b, &sched](auto const& idx) {
               return create_station(b, *sched.stations_[idx]);
             })));
}

Offset<LookupGeoStationResponse> lookup_geo_stations(
    FlatBufferBuilder& b, geo::point_rtree const& station_geo_index,
    schedule const& sched, LookupGeoStationRequest const* req) {
  return CreateLookupGeoStationResponse(
      b, b.CreateVector(utl::to_vec(
             station_geo_index.in_radius({req->pos()->lat(), req->pos()->lng()},
                                         req->min_radius(), req->max_radius()),
             [&b, &sched](auto const& idx) {
               return create_station(b, *sched.stations_[idx]);
             })));
}

}  // namespace motis::lookup
