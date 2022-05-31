#include "motis/paxforecast/alternatives.h"

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <optional>

#include "fmt/format.h"

#include "utl/erase_if.h"
#include "utl/overloaded.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
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

std::uint64_t get_schedule_id(universe const& uv) {
  return uv.uses_default_schedule()
             ? 0ULL
             : static_cast<std::uint64_t>(uv.schedule_res_id_);
}

msg_ptr ontrip_train_query(universe const& uv, schedule const& sched,
                           trip const* trp,
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()), true,
          true, true, get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

msg_ptr ontrip_station_query(universe const& uv, schedule const& sched,
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()), true,
          true, true, get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

msg_ptr pretrip_station_query(universe const& uv, schedule const& sched,
                              unsigned interchange_station_id,
                              time earliest_possible_departure,
                              duration interval_length,
                              unsigned destination_station_id) {
  message_creator fbb;
  auto const interval = Interval{
      motis_to_unixtime(sched, earliest_possible_departure),
      motis_to_unixtime(sched, earliest_possible_departure + interval_length)};
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(
                  fbb,
                  fbb.CreateString(
                      sched.stations_[interchange_station_id]->eva_nr_),
                  fbb.CreateString("")),
              &interval)
              .Union(),
          CreateInputStation(
              fbb,
              fbb.CreateString(
                  sched.stations_[destination_station_id]->eva_nr_),
              fbb.CreateString("")),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()), true,
          true, true, get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

std::string get_cache_key(schedule const& sched,
                          unsigned const destination_station_id,
                          passenger_localization const& localization,
                          duration pretrip_interval_length) {
  if (localization.in_trip()) {
    auto const et = to_extern_trip(sched, localization.in_trip_);
    return fmt::format(
        "{}:{}:{}:{}:trip:{}:{}:{}:{}:{}:{}",
        static_cast<std::uint64_t>(sched.system_time_),
        localization.current_arrival_time_,
        localization.at_station_->eva_nr_.view(),
        sched.stations_.at(destination_station_id)->eva_nr_.view(),
        et.station_id_, et.train_nr_, et.time_, et.target_station_id_,
        et.target_time_, et.line_id_);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    return fmt::format(
        "{}:{}:{}:{}:station{}", static_cast<std::uint64_t>(sched.system_time_),
        localization.current_arrival_time_ + interchange_time,
        localization.at_station_->eva_nr_.view(),
        sched.stations_.at(destination_station_id)->eva_nr_.view(),
        pretrip_interval_length != 0
            ? fmt::format(":{}", pretrip_interval_length)
            : "");
  }
}

msg_ptr send_routing_request(universe const& uv, schedule const& sched,
                             unsigned const destination_station_id,
                             passenger_localization const& localization,
                             duration pretrip_interval_length) {
  msg_ptr query_msg;
  if (localization.in_trip()) {
    query_msg = ontrip_train_query(
        uv, sched, localization.in_trip_, localization.at_station_->index_,
        localization.current_arrival_time_, destination_station_id);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    auto const earliest_possible_departure =
        localization.current_arrival_time_ + interchange_time;
    if (pretrip_interval_length == 0) {
      query_msg = ontrip_station_query(
          uv, sched, localization.at_station_->index_,
          earliest_possible_departure, destination_station_id);
    } else {
      query_msg = pretrip_station_query(
          uv, sched, localization.at_station_->index_,
          earliest_possible_departure, pretrip_interval_length,
          destination_station_id);
    }
  }

  return motis_call(query_msg)->val();
}

msg_ptr get_routing_response(universe const& uv, schedule const& sched,
                             routing_cache& cache,
                             unsigned const destination_station_id,
                             passenger_localization const& localization,
                             bool use_cache, duration pretrip_interval_length) {
  if (use_cache && cache.is_open()) {
    assert(uv.uses_default_schedule());
    auto const cache_key = get_cache_key(sched, destination_station_id,
                                         localization, pretrip_interval_length);
    auto const cache_key_view = std::string_view{
        reinterpret_cast<char const*>(cache_key.data()), cache_key.size()};
    auto msg = cache.get(cache_key_view);
    if (!msg) {
      msg = send_routing_request(uv, sched, destination_station_id,
                                 localization, pretrip_interval_length);
      cache.put(cache_key_view, msg);
    }
    return msg;
  } else {
    return send_routing_request(uv, sched, destination_station_id, localization,
                                pretrip_interval_length);
  }
}

}  // namespace

std::vector<journey> find_alternative_journeys(
    universe const& uv, schedule const& sched, routing_cache& cache,
    unsigned const destination_station_id,
    passenger_localization const& localization, bool use_cache,
    duration pretrip_interval_length) {
  auto const response_msg =
      get_routing_response(uv, sched, cache, destination_station_id,
                           localization, use_cache, pretrip_interval_length);
  auto const response = motis_content(RoutingResponse, response_msg);
  auto alternatives = message_to_journeys(response);
  // TODO(pablo): alternatives without trips?
  utl::erase_if(alternatives, [](journey const& j) {
    return j.stops_.empty() || j.trips_.empty();
  });
  std::sort(
      begin(alternatives), end(alternatives),
      [](journey const& lhs, journey const& rhs) {
        return std::tie(lhs.stops_.back().arrival_.timestamp_, lhs.transfers_) <
               std::tie(rhs.stops_.back().arrival_.timestamp_, rhs.transfers_);
      });
  return alternatives;
}

bool contains_trip(alternative const& alt, extern_trip const& searched_trip) {
  return std::any_of(begin(alt.journey_.trips_), end(alt.journey_.trips_),
                     [&](journey::trip const& jt) {
                       return jt.extern_trip_ == searched_trip;
                     });
}

bool is_recommended(alternative const& alt,
                    measures::trip_recommendation const& m) {
  // TODO(pablo): check interchange stop
  return contains_trip(alt, m.recommended_trip_);
}

void check_measures(
    alternative& alt,
    mcd::vector<measures::measure_variant const*> const& group_measures) {
  for (auto const* mv : group_measures) {
    std::visit(
        utl::overloaded{//
                        [&](measures::trip_recommendation const& m) {
                          if (is_recommended(alt, m)) {
                            alt.is_recommended_ = true;
                          }
                        },
                        [&](measures::trip_load_information const& m) {
                          // TODO(pablo): handle case where load
                          // information for multiple trips in the
                          // journey is available
                          if (contains_trip(alt, m.trip_)) {
                            alt.load_info_ = m.level_;
                          }
                        },
                        [&](measures::trip_load_recommendation const& m) {
                          for (auto const& tll : m.full_trips_) {
                            if (contains_trip(alt, tll.trip_)) {
                              alt.load_info_ = tll.level_;
                            }
                          }
                          for (auto const& tll : m.recommended_trips_) {
                            if (contains_trip(alt, tll.trip_)) {
                              alt.load_info_ = tll.level_;
                              alt.is_recommended_ = true;
                            }
                          }
                        }},
        *mv);
  }
}

std::vector<alternative> find_alternatives(
    universe const& uv, schedule const& sched, routing_cache& cache,
    mcd::vector<measures::measure_variant const*> const& group_measures,
    unsigned const destination_station_id,
    passenger_localization const& localization,
    compact_journey const* remaining_journey, bool use_cache,
    duration pretrip_interval_length) {
  // never use cache for schedule forks
  if (!uv.uses_default_schedule()) {
    use_cache = false;
  }

  // default alternative routing
  auto const journeys = find_alternative_journeys(
      uv, sched, cache, destination_station_id, localization, use_cache,
      pretrip_interval_length);
  auto alternatives = utl::to_vec(journeys, [&](journey const& j) {
    auto const arrival_time = unix_to_motistime(
        sched.schedule_begin_, j.stops_.back().arrival_.timestamp_);
    auto const dur = static_cast<duration>(arrival_time -
                                           localization.current_arrival_time_);
    return alternative{
        j, to_compact_journey(j, sched), arrival_time, dur, j.transfers_, true};
  });

  // TODO(pablo): add additional alternatives for recommended trips (if not
  // already found)
  if (remaining_journey != nullptr) {
    auto recommended_trips_not_found = 0ULL;
    for (auto const* mv : group_measures) {
      std::visit(utl::overloaded{[&](measures::trip_recommendation const& m) {
                   if (!std::any_of(begin(alternatives), end(alternatives),
                                    [&](alternative const& alt) {
                                      return is_recommended(alt, m);
                                    })) {
                     ++recommended_trips_not_found;
                   }
                 }},
                 *mv);
    }
    if (recommended_trips_not_found > 0) {
      LOG(info)
          << recommended_trips_not_found
          << " recommended trips not included in any alternative journeys";
    }
  }

  // TODO(pablo): mark original journey
  for (auto& alt : alternatives) {
    check_measures(alt, group_measures);
  }

  return alternatives;
}

}  // namespace motis::paxforecast
