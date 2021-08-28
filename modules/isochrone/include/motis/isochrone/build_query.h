#pragma once

#include <algorithm>
#include <string>

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/motis_call.h"
#include "geo/latlng.h"
#include "motis/module/message.h"


#include "motis/isochrone/search.h"
#include "motis/isochrone/error.h"


#include "motis/protocol/IsochroneRequest_generated.h"

using namespace geo;
using namespace motis::module;
using namespace motis::lookup;
using namespace motis::osrm;
using namespace motis::ppr;
using namespace flatbuffers;

namespace motis::isochrone {


/*
inline station_node const* get_station_node(schedule const& sched,
                                            InputStation const* input_station) {
  using guesser::StationGuesserResponse;

  std::string station_id;

  if (input_station->id()->Length() != 0) {
    station_id = input_station->id()->str();
  } else {
    module::message_creator b;
    b.create_and_finish(MsgContent_StationGuesserRequest,
                        guesser::CreateStationGuesserRequest(
                            b, 1, b.CreateString(input_station->name()->str()))
                            .Union(),
                        "/guesser");
    auto const msg = motis_call(make_msg(b))->val();
    auto const guesses = motis_content(StationGuesserResponse, msg)->guesses();

    if (guesses->size() == 0) {
      throw std::system_error(error::no_guess_for_station);
    }

    station_id = guesses->Get(0)->id()->str();
  }

  return motis::get_station_node(sched, station_id);
}
*/

inline geo::latlng to_latlng(Position const* pos) {
  return {pos->lat(), pos->lng()};
}

msg_ptr make_geo_request(latlng const& pos, double radius) {
  Position fbs_position{pos.lat_, pos.lng_};
  message_creator mc;
  mc.create_and_finish(
          MsgContent_LookupGeoStationRequest,
          CreateLookupGeoStationRequest(mc, &fbs_position, 0, radius).Union(),
          "/lookup/geo_station");
  return make_msg(mc);
}

msg_ptr make_osrm_request(latlng const& pos,
                          Vector<Offset<Station>> const* stations,
                          std::string const& profile, Direction direction) {
  Position fbs_position{pos.lat_, pos.lng_};
  std::vector<Position> many;
  for (auto const* station : *stations) {
    many.push_back(*station->pos());
  }

  message_creator mc;
  mc.create_and_finish(
          MsgContent_OSRMOneToManyRequest,
          CreateOSRMOneToManyRequest(mc, mc.CreateString(profile), direction,
                                     &fbs_position, mc.CreateVectorOfStructs(many))
                  .Union(),
          "/osrm/one_to_many");
  return make_msg(mc);
}

inline search_query build_query(schedule const& sched,
                                IsochroneRequest const* req) {
  search_query q;
  verify_external_timestamp(sched, req->departure_time());
  auto pos = to_latlng(req->position());
  auto const geo_msg = motis_call(make_geo_request(pos, req->foot_travel_time()))->val();
  auto const geo_resp = motis_content(LookupGeoStationResponse, geo_msg);
  auto const stations = geo_resp->stations();
  /*
  auto const osrm_msg =
          motis_call(make_osrm_request(pos, stations, "foot", Direction_Forward))
                  ->val();
  auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);
*/


  message_creator mc;
  mc.create_and_finish(
          MsgContent_FootRoutingRequest,
          CreateFootRoutingRequest(
                  mc, req->position(),
                  mc.CreateVectorOfStructs(utl::to_vec(
                          *stations, [](auto&& station) { return *station->pos(); })),
                  CreateSearchOptions(mc, mc.CreateString("default"), req->foot_travel_time()), SearchDirection_Forward, false,
                  false, false)
                  .Union(),
          "/ppr/route");
  auto const ppr_msg =
          motis_call(make_msg(mc))
                  ->val();
  auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);

  auto const routes = ppr_resp->routes();
  for (auto i = 0UL; i < routes->size(); ++i) {
    auto const dest_routes = routes->Get(i);
    auto const dest_id = stations->Get(i)->id()->str();
    auto const dest_pos = to_latlng(stations->Get(i)->pos());
    for (auto const& route : *dest_routes->routes()) {
      q.start_stations_.emplace_back(motis::get_station_node(sched, stations->Get(i)->id()->str()), unix_to_motistime(sched, req->departure_time()+route->duration()*60));
    }


  }

  q.interval_begin_ = unix_to_motistime(sched, req->departure_time());
  q.interval_end_ =
      unix_to_motistime(sched, req->departure_time() + req->max_travel_time());

  q.sched_ = &sched;

  return q;
}

}  // namespace motis::isochrone
