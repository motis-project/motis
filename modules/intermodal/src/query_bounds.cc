#include "motis/intermodal/query_bounds.h"

#include "utl/verify.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/access/station_access.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/intermodal/error.h"

using namespace flatbuffers;
using namespace motis::routing;
using namespace motis::module;

namespace motis::intermodal {

inline unixtime get_direct_start_time(Interval const* interval) {
  return interval->begin() + (interval->end() - interval->begin()) / 2;
}

inline geo::latlng get_station_coordinates(InputStation const* s) {
  using lookup::LookupStationLocationResponse;
  module::message_creator b;
  b.create_and_finish(MsgContent_InputStation,
                      motis_copy_table(InputStation, b, s).Union(),
                      "/lookup/station_location");
  auto const msg = motis_call(make_msg(b))->val();
  auto const pos =
      motis_content(LookupStationLocationResponse, msg)->position();
  return {pos->lat(), pos->lng()};
}

query_start parse_query_start(FlatBufferBuilder& fbb,
                              IntermodalRoutingRequest const* req) {
  auto start_station = CreateInputStation(fbb, fbb.CreateString(STATION_START),
                                          fbb.CreateString(STATION_START));
  switch (req->start_type()) {
    case IntermodalStart_IntermodalOntripStart: {
      auto const start =
          reinterpret_cast<IntermodalOntripStart const*>(req->start());
      return {
          Start_OntripStationStart,
          CreateOntripStationStart(fbb, start_station, start->departure_time())
              .Union(),
          {start->position()->lat(), start->position()->lng()},
          start->departure_time(),
          true};
    }

    case IntermodalStart_IntermodalPretripStart: {
      auto const start =
          reinterpret_cast<IntermodalPretripStart const*>(req->start());
      return {Start_PretripStart,
              CreatePretripStart(fbb, start_station, start->interval(),
                                 start->min_connection_count(),
                                 start->extend_interval_earlier(),
                                 start->extend_interval_later())
                  .Union(),
              {start->position()->lat(), start->position()->lng()},
              get_direct_start_time(start->interval()),
              true};
    }

    case IntermodalStart_OntripTrainStart: {
      auto const start =
          reinterpret_cast<OntripTrainStart const*>(req->start());
      return {
          Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb, motis_copy_table(TripId, fbb, start->trip()),
              CreateInputStation(fbb, fbb.CreateString(start->station()->id()),
                                 fbb.CreateString(start->station()->name())),
              start->arrival_time())
              .Union(),
          get_station_coordinates(start->station()), start->arrival_time(),
          false};
    }

    case IntermodalStart_OntripStationStart: {
      auto const start =
          reinterpret_cast<OntripStationStart const*>(req->start());
      return {
          Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString(start->station()->id()),
                                 fbb.CreateString(start->station()->name())),
              start->departure_time())
              .Union(),
          get_station_coordinates(start->station()), start->departure_time(),
          false};
    }

    case IntermodalStart_PretripStart: {
      auto const start = reinterpret_cast<PretripStart const*>(req->start());
      return {
          Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString(start->station()->id()),
                                 fbb.CreateString(start->station()->name())),
              start->interval(), start->min_connection_count(),
              start->extend_interval_earlier(), start->extend_interval_later())
              .Union(),
          get_station_coordinates(start->station()),
          get_direct_start_time(start->interval()), false};
    }

    default: throw utl::fail("invalid query start");
  }
}

query_dest parse_query_dest(FlatBufferBuilder& fbb,
                            IntermodalRoutingRequest const* req) {
  switch (req->destination_type()) {
    case IntermodalDestination_InputStation: {
      return query_dest{
          motis_copy_table(InputStation, fbb, req->destination()), {}, false};
    }

    case IntermodalDestination_InputPosition: {
      auto pos = reinterpret_cast<InputPosition const*>(req->destination());
      auto end_station = CreateInputStation(fbb, fbb.CreateString(STATION_END),
                                            fbb.CreateString(STATION_END));
      return {end_station, {pos->lat(), pos->lng()}, true};
    }

    default: throw utl::fail("invalid query dest");
  }
}

}  // namespace motis::intermodal
