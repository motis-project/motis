#include "motis/paxforecast/alternatives.h"

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <iostream>
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
#include "motis/core/journey/print_journey.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/message.h"

#include "motis/paxmon/debug.h"
#include "motis/paxmon/loader/motis_journeys/to_compact_journey.h"
#include "motis/paxmon/localization.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::logging;
using namespace motis::routing;
using namespace motis::paxmon;

namespace motis::paxforecast {

namespace {

auto const constexpr USE_START_FOOTPATHS = true;

std::uint64_t get_schedule_id(universe const& uv) {
  return uv.uses_default_schedule()
             ? 0ULL
             : static_cast<std::uint64_t>(uv.schedule_res_id_);
}

msg_ptr ontrip_train_query(universe const& uv, schedule const& sched,
                           trip const* trp,
                           unsigned const first_possible_interchange_station_id,
                           time const first_possible_interchange_arrival,
                           unsigned const destination_station_id,
                           bool const allow_start_metas,
                           bool const allow_dest_metas) {
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
          allow_start_metas, allow_dest_metas, USE_START_FOOTPATHS,
          get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

msg_ptr ontrip_station_query(universe const& uv, schedule const& sched,
                             unsigned const interchange_station_id,
                             time const earliest_possible_departure,
                             unsigned const destination_station_id,
                             bool const allow_start_metas,
                             bool const allow_dest_metas) {
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
          allow_start_metas, allow_dest_metas, USE_START_FOOTPATHS,
          get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

msg_ptr pretrip_station_query(universe const& uv, schedule const& sched,
                              unsigned interchange_station_id,
                              time const earliest_possible_departure,
                              duration const interval_length,
                              unsigned const destination_station_id,
                              bool const allow_start_metas,
                              bool const allow_dest_metas) {
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()),
          allow_start_metas, allow_dest_metas, USE_START_FOOTPATHS,
          get_schedule_id(uv))
          .Union(),
      "/routing");
  return make_msg(fbb);
}

std::string get_cache_key(schedule const& sched,
                          unsigned const destination_station_id,
                          passenger_localization const& localization,
                          alternative_routing_options const& options) {
  if (localization.in_trip()) {
    auto const et = to_extern_trip(sched, localization.in_trip_);
    return fmt::format(
        "{}:{}:{}:{}:trip:{}:{}:{}:{}:{}:{}:{}:{}",
        static_cast<std::uint64_t>(sched.system_time_),
        localization.current_arrival_time_,
        localization.at_station_->eva_nr_.view(),
        sched.stations_.at(destination_station_id)->eva_nr_.view(),
        et.station_id_, et.train_nr_, et.time_, et.target_station_id_,
        et.target_time_, et.line_id_, options.allow_start_metas_,
        options.allow_dest_metas_);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    return fmt::format(
        "{}:{}:{}:{}:station{}:{}:{}",
        static_cast<std::uint64_t>(sched.system_time_),
        localization.current_arrival_time_ + interchange_time,
        localization.at_station_->eva_nr_.view(),
        sched.stations_.at(destination_station_id)->eva_nr_.view(),
        options.pretrip_interval_length_ != 0
            ? fmt::format(":{}", options.pretrip_interval_length_)
            : "",
        options.allow_start_metas_, options.allow_dest_metas_);
  }
}

msg_ptr send_routing_request(universe const& uv, schedule const& sched,
                             unsigned const destination_station_id,
                             passenger_localization const& localization,
                             alternative_routing_options const& options) {
  msg_ptr query_msg;
  if (localization.in_trip()) {
    query_msg = ontrip_train_query(
        uv, sched, localization.in_trip_, localization.at_station_->index_,
        localization.current_arrival_time_, destination_station_id,
        options.allow_start_metas_, options.allow_dest_metas_);
  } else {
    auto const interchange_time =
        localization.first_station_
            ? 0
            : sched.stations_.at(localization.at_station_->index_)
                  ->transfer_time_;
    auto const earliest_possible_departure =
        localization.current_arrival_time_ + interchange_time;
    if (options.pretrip_interval_length_ == 0) {
      query_msg = ontrip_station_query(
          uv, sched, localization.at_station_->index_,
          earliest_possible_departure, destination_station_id,
          options.allow_start_metas_, options.allow_dest_metas_);
    } else {
      query_msg = pretrip_station_query(
          uv, sched, localization.at_station_->index_,
          earliest_possible_departure, options.pretrip_interval_length_,
          destination_station_id, options.allow_start_metas_,
          options.allow_dest_metas_);
    }
  }

  return motis_call(query_msg)->val();
}

msg_ptr get_routing_response(universe const& uv, schedule const& sched,
                             routing_cache& cache,
                             unsigned const destination_station_id,
                             passenger_localization const& localization,
                             alternative_routing_options const& options) {
  if (options.use_cache_ && cache.is_open()) {
    assert(uv.uses_default_schedule());
    auto const cache_key =
        get_cache_key(sched, destination_station_id, localization, options);
    auto const cache_key_view = std::string_view{
        reinterpret_cast<char const*>(cache_key.data()), cache_key.size()};
    auto msg = cache.get(cache_key_view);
    if (!msg) {
      msg = send_routing_request(uv, sched, destination_station_id,
                                 localization, options);
      cache.put(cache_key_view, msg);
    }
    return msg;
  } else {
    return send_routing_request(uv, sched, destination_station_id, localization,
                                options);
  }
}

}  // namespace

std::vector<journey> find_alternative_journeys(
    universe const& uv, schedule const& sched, routing_cache& cache,
    unsigned const destination_station_id,
    passenger_localization const& localization,
    alternative_routing_options const& options, bool debug) {
  auto const response_msg = get_routing_response(
      uv, sched, cache, destination_station_id, localization, options);
  auto const response = motis_content(RoutingResponse, response_msg);
  auto alternatives = message_to_journeys(response);

  if (debug) {
    std::cout << "find_alternative_journeys debug:" << std::endl;
    std::cout << "destination: "
              << sched.stations_.at(destination_station_id)->name_
              << "\nlocalization: next station: "
              << localization.at_station_->name_
              << (localization.in_trip() ? " (in trip)" : "") << " at "
              << format_time(localization.current_arrival_time_) << std::endl;
    for (auto const& j : alternatives) {
      print_journey(j);
      std::cout << std::endl;
    }
    std::cout << response_msg->to_json() << std::endl;
  }

  utl::erase_if(alternatives,
                [](journey const& j) { return j.stops_.size() < 2; });
  std::sort(
      begin(alternatives), end(alternatives),
      [](journey const& lhs, journey const& rhs) {
        return std::tie(lhs.stops_.back().arrival_.timestamp_, lhs.transfers_) <
               std::tie(rhs.stops_.back().arrival_.timestamp_, rhs.transfers_);
      });
  return alternatives;
}

std::vector<alternative> find_alternatives(
    universe const& uv, schedule const& sched, routing_cache& cache,
    unsigned const destination_station_id,
    passenger_localization const& localization,
    alternative_routing_options options) {
  // never use cache for schedule forks
  if (!uv.uses_default_schedule()) {
    options.use_cache_ = false;
  }

  if (!localization.first_station_) {
    options.allow_start_metas_ = false;
  }

  auto const debug = false;

  // default alternative routing
  auto const journeys = find_alternative_journeys(
      uv, sched, cache, destination_station_id, localization, options, debug);
  auto alternatives = utl::to_vec(journeys, [&](journey const& j) {
    auto const arrival_time = unix_to_motistime(
        sched.schedule_begin_, j.stops_.back().arrival_.timestamp_);
    auto const dur = static_cast<duration>(arrival_time -
                                           localization.current_arrival_time_);
    return alternative{.journey_ = j,
                       .compact_journey_ = to_compact_journey(j, sched),
                       .arrival_time_ = arrival_time,
                       .duration_ = dur,
                       .transfers_ = j.transfers_};
  });

  if (alternatives.empty() &&
      (options.allow_start_metas_ || options.allow_dest_metas_)) {
    if (std::any_of(begin(localization.at_station_->equivalent_),
                    end(localization.at_station_->equivalent_),
                    [&](auto const& eq) {
                      return eq->index_ == destination_station_id;
                    })) {
      // reached destination meta station
      auto alt = alternative{};
      alt.arrival_time_ = localization.current_arrival_time_;
      alternatives.emplace_back(alt);
    }
  }

  return alternatives;
}

std::uint32_t alternatives_set::add_request(
    passenger_localization const& localization,
    unsigned const destination_station_id) {
  auto const key = mcd::pair{localization, destination_station_id};
  if (auto const it = request_key_to_idx_.find(key);
      it != end(request_key_to_idx_)) {
    return it->second;
  } else {
    auto const idx = static_cast<std::uint32_t>(requests_.size());
    requests_.emplace_back(alternatives_request{
        .localization_ = localization,
        .destination_station_id_ = destination_station_id});
    request_key_to_idx_[key] = idx;
    return idx;
  }
}

void alternatives_set::find(universe const& uv, schedule const& sched,
                            routing_cache& cache,
                            alternative_routing_options const& options) {
  auto futures = utl::to_vec(requests_, [&](alternatives_request& req) {
    return spawn_job_void([&uv, &sched, &cache, &req, &options] {
      req.alternatives_ =
          find_alternatives(uv, sched, cache, req.destination_station_id_,
                            req.localization_, options);
    });
  });
  ctx::await_all(futures);
  cache.sync();
}

}  // namespace motis::paxforecast
