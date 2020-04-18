#include "motis/intermodal/intermodal.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/intermodal/direct_connections.h"
#include "motis/intermodal/error.h"
#include "motis/intermodal/mumo_edge.h"
#include "motis/intermodal/query_bounds.h"
#include "motis/intermodal/statistics.h"

#include "motis/protocol/Message_generated.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::lookup;
using namespace motis::osrm;
using namespace motis::routing;
using namespace motis::revise;

namespace motis::intermodal {

intermodal::intermodal() : module("Intermodal Options", "intermodal") {
  param(router_, "router", "routing module");
  param(revise_, "revise", "revise connections");
}

intermodal::~intermodal() = default;

void intermodal::init(motis::module::registry& r) {
  r.register_op("/intermodal", [this](msg_ptr const& m) { return route(m); });
  if (router_.empty()) {
    router_ = "/routing";
  } else if (router_[0] != '/') {
    router_ = "/" + router_;
  }
  r.subscribe("/init", [this]() { ppr_profiles_.update(); });
}

std::vector<Offset<Connection>> revise_connections(
    std::vector<journey> const& journeys, statistics& stats,
    message_creator& mc) {
  MOTIS_START_TIMING(revise_timing);
  message_creator rmc;
  rmc.create_and_finish(
      MsgContent_ReviseRequest,
      CreateReviseRequest(
          rmc, rmc.CreateVector(utl::to_vec(
                   journeys,
                   [&](journey const& j) { return to_connection(rmc, j); })))
          .Union(),
      "/revise");
  auto const res = motis_call(make_msg(rmc))->val();
  auto result = utl::to_vec(*motis_content(ReviseResponse, res)->connections(),
                            [&mc](Connection const* con) {
                              return motis_copy_table(Connection, mc, con);
                            });
  MOTIS_STOP_TIMING(revise_timing);
  stats.revise_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(revise_timing));
  return result;
}

struct parking_patch {
  parking_patch(mumo_edge const* e, unsigned transport_from,
                unsigned transport_to)
      : e_(e), from_(transport_from), to_(transport_to) {}

  mumo_edge const* e_{};
  unsigned from_{};
  unsigned to_{};
};

struct split_transport_result {
  journey::stop& parking_stop_;
  journey::transport& first_transport_;
  journey::transport& second_transport_;
};

split_transport_result split_transport(journey& j,
                                       std::vector<parking_patch>& patches,
                                       journey::transport& transport) {
  utl::verify(transport.to_ == transport.from_ + 1, "invalid transport");
  auto const transport_idx = std::distance(
      begin(j.transports_),
      std::find_if(
          begin(j.transports_), end(j.transports_), [&](auto const& t) {
            return t.from_ == transport.from_ && t.to_ == transport.to_;
          }));
  auto const new_stop_idx = transport.to_;
  auto& new_stop = *j.stops_.emplace(j.stops_.begin() + new_stop_idx);

  auto const update_idx = [&](unsigned& i) {
    if (i >= new_stop_idx) {
      ++i;
    }
  };

  auto const update_indices = [&](auto& v) {
    for (auto& e : v) {
      update_idx(e.from_);
      update_idx(e.to_);
    }
  };

  update_indices(j.transports_);
  update_indices(j.trips_);
  update_indices(j.attributes_);
  update_indices(j.free_texts_);
  update_indices(j.problems_);
  update_indices(patches);

  j.transports_.insert(begin(j.transports_) + transport_idx,
                       j.transports_[transport_idx]);

  auto& t1 = j.transports_[transport_idx];
  auto& t2 = j.transports_[transport_idx + 1];

  t1.to_--;
  t2.from_++;

  return {new_stop, t1, t2};
}

std::string get_parking_station(int index) {
  switch (index) {
    case 0: return STATION_VIA0;
    case 1: return STATION_VIA1;
    default: throw std::system_error(error::parking_edge_error);
  }
}

void apply_parking_patches(journey& j, std::vector<parking_patch>& patches) {
  auto const get_transport = [&](unsigned const from,
                                 unsigned const to) -> journey::transport& {
    for (auto& t : j.transports_) {
      if (t.from_ == from && t.to_ == to) {
        return t;
      }
    }
    throw std::system_error(error::parking_edge_error);
  };

  auto const is_virtual_station = [](journey::stop const& s) {
    return s.name_ == STATION_START || s.name_ == STATION_END;
  };

  auto parking_idx = 0;
  for (auto& p : patches) {
    auto t = get_transport(p.from_, p.to_);
    auto const car_first = is_virtual_station(j.stops_[p.from_]);

    auto const first_edge_duration =
        (car_first ? p.e_->car_parking_->car_duration_
                   : p.e_->car_parking_->foot_duration_) *
        60;

    auto str = split_transport(j, patches, t);
    auto const parking_station = get_parking_station(parking_idx++);
    str.parking_stop_.eva_no_ = parking_station;
    str.parking_stop_.name_ = parking_station;
    str.parking_stop_.lat_ = p.e_->car_parking_->parking_pos_.lat_;
    str.parking_stop_.lng_ = p.e_->car_parking_->parking_pos_.lng_;
    str.parking_stop_.arrival_.valid_ = true;
    str.parking_stop_.arrival_.timestamp_ =
        j.stops_[p.from_].departure_.timestamp_ + first_edge_duration;
    str.parking_stop_.arrival_.schedule_timestamp_ =
        j.stops_[p.from_].departure_.schedule_timestamp_ + first_edge_duration;
    str.parking_stop_.arrival_.timestamp_reason_ =
        j.stops_[p.from_].departure_.timestamp_reason_;
    str.parking_stop_.departure_ = str.parking_stop_.arrival_;

    auto& car_transport =
        car_first ? str.first_transport_ : str.second_transport_;
    auto& foot_transport =
        car_first ? str.second_transport_ : str.first_transport_;

    car_transport.mumo_type_ = to_string(mumo_type::CAR);
    foot_transport.mumo_type_ = to_string(mumo_type::FOOT);
  }
}

msg_ptr postprocess_response(msg_ptr const& response_msg,
                             query_start const& q_start,
                             query_dest const& q_dest,
                             IntermodalRoutingRequest const* req,
                             std::vector<mumo_edge const*> const& edge_mapping,
                             statistics& stats, bool const revise,
                             std::vector<stats_category> const& mumo_stats,
                             ppr_profiles const& profiles) {
  auto const dir = req->search_dir();
  auto routing_response = motis_content(RoutingResponse, response_msg);
  auto journeys = message_to_journeys(routing_response);

  message_creator mc;
  for (auto& journey : journeys) {
    auto& stops = journey.stops_;
    if (stops.size() < 2) {
      continue;
    }

    if (q_start.is_intermodal_) {
      auto& start = (dir == SearchDir_Forward) ? stops.front() : stops.back();
      start.lat_ = q_start.pos_.lat_;
      start.lng_ = q_start.pos_.lng_;
    }

    if (q_dest.is_intermodal_) {
      auto& dest = (dir == SearchDir_Forward) ? stops.back() : stops.front();
      dest.lat_ = q_dest.pos_.lat_;
      dest.lng_ = q_dest.pos_.lng_;
    }

    std::vector<parking_patch> patches;

    for (auto& t : journey.transports_) {
      if (!t.is_walk_ || t.mumo_id_ < 0) {
        continue;
      }

      auto const e = edge_mapping.at(static_cast<std::size_t>(t.mumo_id_));
      t.mumo_type_ = to_string(e->type_);
      t.mumo_id_ = e->id_;

      if (e->type_ == mumo_type::CAR_PARKING && e->car_parking_) {
        if (!e->car_parking_->uses_car_) {
          t.mumo_type_ = to_string(mumo_type::FOOT);
          continue;
        }
        patches.emplace_back(e, t.from_, t.to_);
      }
    }

    if (!patches.empty()) {
      apply_parking_patches(journey, patches);
    }
  }

  MOTIS_START_TIMING(direct_connection_timing);
  auto const direct = get_direct_connections(q_start, q_dest, req, profiles);
  stats.dominated_by_direct_connection_ =
      remove_dominated_journeys(journeys, direct);
  add_direct_connections(journeys, direct, q_start, q_dest, req);
  MOTIS_STOP_TIMING(direct_connection_timing);
  stats.direct_connection_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(direct_connection_timing));

  utl::erase_if(journeys, [](journey const& j) { return j.stops_.empty(); });
  std::sort(
      begin(journeys), end(journeys), [](journey const& a, journey const& b) {
        return std::make_pair(a.stops_.front().departure_.schedule_timestamp_,
                              a.stops_.back().arrival_.schedule_timestamp_) <
               std::make_pair(b.stops_.front().departure_.schedule_timestamp_,
                              b.stops_.back().arrival_.schedule_timestamp_);
      });

  auto const connections = revise
                               ? revise_connections(journeys, stats, mc)
                               : utl::to_vec(journeys, [&mc](journey const& j) {
                                   return to_connection(mc, j);
                                 });
  auto all_stats = utl::to_vec(*routing_response->statistics(),
                               [&](Statistics const* stats) {
                                 return motis_copy_table(Statistics, mc, stats);
                               });
  for (auto const& s : mumo_stats) {
    all_stats.emplace_back(to_fbs(mc, s));
  }
  all_stats.emplace_back(to_fbs(mc, to_stats_category("intermodal", stats)));
  mc.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          mc, mc.CreateVectorOfSortedTables(&all_stats),
          mc.CreateVector(connections), routing_response->interval_begin(),
          routing_response->interval_end(),
          mc.CreateVector(utl::to_vec(
              direct,
              [&mc](direct_connection const& c) { return to_fbs(mc, c); })))
          .Union());

  return make_msg(mc);
}

msg_ptr empty_response(statistics& stats, schedule const& sched) {
  auto const schedule_begin = SCHEDULE_OFFSET_MINUTES;
  auto const schedule_end =
      static_cast<time>((sched.schedule_end_ - sched.schedule_begin_) / 60);
  message_creator mc;
  std::vector<Offset<Statistics>> all_stats{
      to_fbs(mc, to_stats_category("intermodal", stats))};
  mc.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          mc, mc.CreateVectorOfSortedTables(&all_stats),
          mc.CreateVector(std::vector<Offset<Connection>>{}),
          motis_to_unixtime(sched, schedule_begin),
          motis_to_unixtime(sched, schedule_end),
          mc.CreateVector(std::vector<Offset<DirectConnection>>{}))
          .Union());

  return make_msg(mc);
}

msg_ptr intermodal::route(msg_ptr const& msg) {
  auto const req = motis_content(IntermodalRoutingRequest, msg);
  message_creator mc;
  statistics stats{};

  auto const& sched = get_schedule();
  auto const start = parse_query_start(mc, req, sched);
  auto const dest = parse_query_dest(mc, req, sched);

  stats.linear_distance_ =
      static_cast<uint64_t>(distance(start.pos_, dest.pos_));

  auto appender = [](auto& vec, auto const& from, auto const& to,
                     auto const& from_pos, auto const& to_pos, auto const dur,
                     auto const accessibility, mumo_type const type,
                     int const id) -> mumo_edge& {
    return vec.emplace_back(from, to, from_pos, to_pos, dur, accessibility,
                            type, id);
  };

  MOTIS_START_TIMING(mumo_edge_timing);
  std::vector<mumo_edge> deps;
  std::vector<mumo_edge> arrs;
  std::vector<stats_category> mumo_stats;
  std::mutex mumo_stats_mutex;

  auto const mumo_stats_appender = [&](stats_category&& s) {
    std::lock_guard guard(mumo_stats_mutex);
    mumo_stats.emplace_back(s);
  };

  std::vector<ctx::future_ptr<ctx_data, void>> futures;

  using namespace std::placeholders;
  if (req->search_dir() == SearchDir_Forward) {
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      STATION_START, _1, start.pos_, _2, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             _1, STATION_END, _2, dest.pos_, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_);
      }));
    }
  } else {
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      _1, STATION_START, _2, start.pos_, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             STATION_END, _1, dest.pos_, _2, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_);
      }));
    }
  }

  ctx::await_all(futures);
  MOTIS_STOP_TIMING(mumo_edge_timing);

  stats.start_edges_ = deps.size();
  stats.destination_edges_ = arrs.size();
  stats.mumo_edge_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(mumo_edge_timing));

  if ((start.is_intermodal_ && deps.empty()) ||
      (dest.is_intermodal_ && arrs.empty())) {
    return empty_response(stats, sched);
  }

  //  remove_intersection(deps, arrs, start.pos_, dest.pos_, req->search_dir());
  std::vector<mumo_edge const*> edge_mapping;
  auto edges = write_edges(mc, deps, arrs, edge_mapping);

  auto const router = ((req->search_type() == SearchType_Default ||
                        req->search_type() == SearchType_Accessibility) &&
                       start.start_type_ != Start_OntripTrainStart)
                          ? router_
                          : "/routing";

  mc.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(mc, start.start_type_, start.start_, dest.station_,
                           req->search_type(), req->search_dir(),
                           mc.CreateVector(std::vector<Offset<Via>>{}),
                           mc.CreateVector(edges))
          .Union(),
      router);

  MOTIS_START_TIMING(routing_timing);
  auto resp = motis_call(make_msg(mc))->val();
  MOTIS_STOP_TIMING(routing_timing);
  stats.routing_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(routing_timing));
  return postprocess_response(resp, start, dest, req, edge_mapping, stats,
                              revise_, mumo_stats, ppr_profiles_);
}

}  // namespace motis::intermodal
