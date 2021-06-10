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

#include "motis/isochrone/error.h"
#include "motis/isochrone/search.h"

#include "motis/protocol/IsochroneRequest_generated.h"

namespace motis::isochrone {

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

inline std::pair<node const*, light_connection const*> get_ontrip_train_start(
    schedule const& sched, TripId const* trip, station_node const* station,
    time arrival_time) {
  auto const stops = access::stops(from_fbs(sched, trip));
  auto const stop_it = std::find_if(
      begin(stops), end(stops), [&](access::trip_stop const& stop) {
        return stop.has_arrival() &&
               stop.get_route_node()->station_node_ == station &&
               stop.arr_lcon().a_time_ == arrival_time;
      });
  if (stop_it == end(stops)) {
    throw std::system_error(error::event_not_found);
  }
  return {(*stop_it).get_route_node(), &(*stop_it).arr_lcon()};
}

inline search_query build_query(schedule const& sched,
                                IsochroneRequest const* req) {
  search_query q;
  verify_external_timestamp(sched, req->departure_time());
  auto s = req->station();
  q.from_ = get_station_node(sched, req->station());
  q.interval_begin_ = unix_to_motistime(sched, req->departure_time());
  q.interval_end_ =
      unix_to_motistime(sched, req->departure_time() + req->max_travel_time());
  /
  q.sched_ = &sched;

  return q;
}

}  // namespace motis::isochrone
