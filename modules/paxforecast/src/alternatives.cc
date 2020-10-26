#include "motis/paxforecast/alternatives.h"

#include <cstdint>
#include <algorithm>
#include <optional>

#include "cista/serialization.h"

#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/paxmon/loader/journeys/to_compact_journey.h"
#include "motis/paxmon/localization.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::logging;
using namespace motis::routing;
using namespace motis::paxmon;

namespace motis::paxforecast {

namespace {

msg_ptr ontrip_train_query(schedule const& sched, trip const* trp,
                           unsigned first_possible_interchange_station_id,
                           time first_possible_interchange_arrival,
                           unsigned destination_station_id) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripTrainStart,
          CreateOntripTrainStart(
              fbb, to_fbs(sched, fbb, trp),
              CreateInputStation(
                  fbb,
                  fbb.CreateString(
                      sched.stations_[first_possible_interchange_station_id]
                          ->eva_nr_),
                  fbb.CreateString("")),
              motis_to_unixtime(sched, first_possible_interchange_arrival))
              .Union(),
          CreateInputStation(
              fbb,
              fbb.CreateString(
                  sched.stations_[destination_station_id]->eva_nr_),
              fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

msg_ptr ontrip_station_query(schedule const& sched,
                             unsigned interchange_station_id,
                             time earliest_possible_departure,
                             unsigned destination_station_id) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_OntripStationStart,
          CreateOntripStationStart(
              fbb,
              CreateInputStation(
                  fbb,
                  fbb.CreateString(
                      sched.stations_[interchange_station_id]->eva_nr_),
                  fbb.CreateString("")),
              motis_to_unixtime(sched, earliest_possible_departure))
              .Union(),
          CreateInputStation(
              fbb,
              fbb.CreateString(
                  sched.stations_[destination_station_id]->eva_nr_),
              fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

struct cache_key_t {
  std::uint64_t system_time_{};
  time next_station_time_{};
  bool ontrip_{};
  mcd::string next_station_eva_{};
  mcd::string dest_station_eva_{};
  extern_trip trip_{};
};

cista::byte_buf get_cache_key(schedule const& sched,
                              unsigned const destination_station_id,
                              passenger_localization const& localization) {
  auto key = cache_key_t{static_cast<std::uint64_t>(sched.system_time_),
                         localization.current_arrival_time_,
                         localization.in_trip(),
                         localization.at_station_->eva_nr_,
                         sched.stations_.at(destination_station_id)->eva_nr_,
                         {}};
  if (localization.in_trip()) {
    key.trip_ = to_extern_trip(sched, localization.in_trip_);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    key.next_station_time_ += interchange_time;
  }
  return cista::serialize(key);
}

msg_ptr send_routing_request(schedule const& sched,
                             unsigned const destination_station_id,
                             passenger_localization const& localization) {
  msg_ptr query_msg;
  if (localization.in_trip()) {
    query_msg = ontrip_train_query(
        sched, localization.in_trip_, localization.at_station_->index_,
        localization.current_arrival_time_, destination_station_id);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    query_msg = ontrip_station_query(
        sched, localization.at_station_->index_,
        localization.current_arrival_time_ + interchange_time,
        destination_station_id);
  }

  return motis_call(query_msg)->val();
}

msg_ptr get_routing_response(schedule const& sched,
                             unsigned const destination_station_id,
                             passenger_localization const& localization,
                             routing_cache& cache) {
  if (cache.is_open()) {
    auto const cache_key =
        get_cache_key(sched, destination_station_id, localization);
    auto const cache_key_view = std::string_view{
        reinterpret_cast<char const*>(cache_key.data()), cache_key.size()};
    auto msg = cache.get(cache_key_view);
    if (!msg) {
      msg = send_routing_request(sched, destination_station_id, localization);
      cache.put(cache_key_view, msg);
    }
    return msg;
  } else {
    return send_routing_request(sched, destination_station_id, localization);
  }
}

}  // namespace

std::vector<alternative> find_alternatives(
    schedule const& sched, unsigned const destination_station_id,
    passenger_localization const& localization, routing_cache& cache) {

  auto const response_msg =
      get_routing_response(sched, destination_station_id, localization, cache);
  auto const response = motis_content(RoutingResponse, response_msg);
  auto alternatives = message_to_journeys(response);
  std::sort(
      begin(alternatives), end(alternatives),
      [](journey const& lhs, journey const& rhs) {
        return std::tie(lhs.stops_.back().arrival_.timestamp_, lhs.transfers_) <
               std::tie(rhs.stops_.back().arrival_.timestamp_, rhs.transfers_);
      });
  return utl::to_vec(alternatives, [&](journey const& j) {
    auto const arrival_time = unix_to_motistime(
        sched.schedule_begin_, j.stops_.back().arrival_.timestamp_);
    auto const dur = static_cast<duration>(arrival_time -
                                           localization.current_arrival_time_);
    return alternative{j, to_compact_journey(j, sched), arrival_time, dur,
                       j.transfers_};
  });
}

}  // namespace motis::paxforecast
