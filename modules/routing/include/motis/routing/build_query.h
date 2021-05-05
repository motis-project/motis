#pragma once

#include <algorithm>
#include <string>

#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/motis_call.h"

#include "motis/routing/error.h"
#include "motis/routing/search.h"

#include "motis/protocol/RoutingRequest_generated.h"

namespace motis::routing {

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
                                RoutingRequest const* req) {
  search_query q;

  utl::verify_ex(!req->include_equivalent(),
                 std::system_error{error::include_equivalent_not_supported});

  switch (req->start_type()) {
    case Start_PretripStart: {
      auto const start = reinterpret_cast<PretripStart const*>(req->start());
      verify_external_timestamp(sched, start->interval()->begin());
      verify_external_timestamp(sched, start->interval()->end());

      q.from_ = get_station_node(sched, start->station());
      q.interval_begin_ = unix_to_motistime(sched, start->interval()->begin());
      q.interval_end_ = unix_to_motistime(sched, start->interval()->end());
      q.min_journey_count_ = start->min_connection_count();
      q.extend_interval_earlier_ = start->extend_interval_earlier();
      q.extend_interval_later_ = start->extend_interval_later();
      q.use_start_metas_ = req->use_start_metas();
      q.use_dest_metas_ = req->use_dest_metas();
      q.use_start_footpaths_ = req->use_start_footpaths();
      break;
    }

    case Start_OntripStationStart: {
      auto start = reinterpret_cast<OntripStationStart const*>(req->start());
      verify_external_timestamp(sched, start->departure_time());

      q.from_ = get_station_node(sched, start->station());
      q.interval_begin_ = unix_to_motistime(sched, start->departure_time());
      q.interval_end_ = INVALID_TIME;
      q.use_dest_metas_ = req->use_dest_metas();
      q.use_start_footpaths_ = req->use_start_footpaths();
      break;
    }

    case Start_OntripTrainStart: {
      auto start = reinterpret_cast<OntripTrainStart const*>(req->start());
      auto const ontrip_start = get_ontrip_train_start(
          sched, start->trip(), get_station_node(sched, start->station()),
          unix_to_motistime(sched, start->arrival_time()));
      q.interval_begin_ = unix_to_motistime(sched, start->arrival_time());
      q.from_ = ontrip_start.first;
      q.lcon_ = ontrip_start.second;
      q.interval_end_ = INVALID_TIME;
      q.use_dest_metas_ = req->use_dest_metas();
      q.use_start_footpaths_ = req->use_start_footpaths();
      break;
    }

    case Start_NONE: assert(false);
  }

  q.sched_ = &sched;
  q.to_ = get_station_node(sched, req->destination());
  q.query_edges_ = create_additional_edges(req->additional_edges(), sched);

  // TODO(Felix Guendling) remove when more edge types are supported
  if (req->search_dir() == SearchDir_Backward &&
      std::any_of(begin(q.query_edges_), end(q.query_edges_),
                  [](edge const& e) { return e.type() != edge::MUMO_EDGE; })) {
    throw std::system_error(error::edge_type_not_supported);
  }

  return q;
}

}  // namespace motis::routing
