#include "motis/intermodal/intermodal.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/intermodal/direct_connections.h"
#include "motis/intermodal/error.h"
#include "motis/intermodal/eval/commands.h"
#include "motis/intermodal/metrics.h"
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
  param(timeout_, "timeout", "routing timeout in seconds (0 = no timeout)");
}

intermodal::~intermodal() = default;

void intermodal::reg_subc(motis::module::subc_reg& r) {
  r.register_cmd("compare", "print difference between results", eval::compare);
}

void intermodal::init(motis::module::registry& r) {
  auto prometheus_registry =
      get_shared_data<std::shared_ptr<prometheus::Registry>>(
          to_res_id(global_res_id::METRICS));

  auto& request_counter_family =
      prometheus::BuildCounter()
          .Name("intermodal_requests_total")
          .Help("Number of intermodal routing requests")
          .Register(*prometheus_registry);

  auto& mode_counter_family = prometheus::BuildCounter()
                                  .Name("intermodal_modes_total")
                                  .Help("Number of intermodal routing requests")
                                  .Register(*prometheus_registry);

  auto const time_buckets = prometheus::Histogram::BucketBoundaries{
      .05, .1,  .25, .5,  .75,  1.0,  2.0,  3.0,  4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 45.0, 60.0};

  metrics_ = std::make_unique<metrics>(metrics{
      .registry_ = *prometheus_registry,
      .fwd_requests_ = request_counter_family.Add({{"direction", "forward"}}),
      .bwd_requests_ = request_counter_family.Add({{"direction", "backward"}}),
      .foot_modes_ = mode_counter_family.Add({{"mode", "foot"}}),
      .foot_ppr_modes_ = mode_counter_family.Add({{"mode", "foot_ppr"}}),
      .bike_modes_ = mode_counter_family.Add({{"mode", "bike"}}),
      .car_modes_ = mode_counter_family.Add({{"mode", "car"}}),
      .car_parking_modes_ = mode_counter_family.Add({{"mode", "car_parking"}}),
      .gbfs_modes_ = mode_counter_family.Add({{"mode", "gbfs"}}),
      .mumo_edges_time_ =
          prometheus::BuildHistogram()
              .Name("intermodal_mumo_edges_time_seconds")
              .Help("Total time to calculate mumo edges per routing request")
              .Register(*prometheus_registry)
              .Add({}, time_buckets),
      .total_time_ = prometheus::BuildHistogram()
                         .Name("intermodal_total_time_seconds")
                         .Help("Total time per intermodal routing request")
                         .Register(*prometheus_registry)
                         .Add({}, time_buckets),
  });

  r.register_op("/intermodal", [this](msg_ptr const& m) { return route(m); },
                {});
  if (router_.empty()) {
    router_ = "/routing";
  } else if (router_[0] != '/') {
    router_ = "/" + router_;
  }
  r.subscribe("/init", [this]() { ppr_profiles_.update(); }, {});
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
    case 2: return STATION_VIA2;
    case 3: return STATION_VIA3;
    default: throw std::system_error(error::parking_edge_error);
  }
}

journey::transport& get_transport(journey& j, unsigned const from,
                                  unsigned const to) {
  for (auto& t : j.transports_) {
    if (t.from_ == from && t.to_ == to) {
      return t;
    }
  }
  throw std::system_error(error::parking_edge_error);
}

bool is_virtual_station(journey::stop const& s) {
  return s.name_ == STATION_START || s.name_ == STATION_END;
}

void apply_parking_patches(journey& j, std::vector<parking_patch>& patches) {
  auto parking_idx = 0;
  for (auto& p : patches) {
    auto t = get_transport(j, p.from_, p.to_);
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

void apply_gbfs_patches(journey& j, std::vector<parking_patch>& patches) {
  for (auto const& p : patches) {
    // station bike:
    // replace: X --walk[type:gbfs]--> P
    // to: X --walk--> (SX) --bike--> (SP) --walk--> P
    // replace: P -->walk[type:gbfs]--> X
    // to: P --walk--> (SP) --bike--> (SX) --walk--> X
    if (std::holds_alternative<gbfs_edge::station_bike>(p.e_->gbfs_->bike_)) {
      auto const& s = std::get<gbfs_edge::station_bike>(p.e_->gbfs_->bike_);

      auto& t = get_transport(j, p.from_, p.to_);
      auto str1 = split_transport(j, patches, t);
      split_transport(j, patches, str1.second_transport_);

      auto& s1 = j.stops_.at(p.from_ + 1);
      s1.eva_no_ = s.from_station_id_;
      s1.name_ = s.from_station_name_;
      s1.lat_ = s.from_station_pos_.lat_;
      s1.lng_ = s.from_station_pos_.lng_;
      s1.arrival_.valid_ = true;
      s1.arrival_.timestamp_ =
          j.stops_[p.from_].departure_.timestamp_ + s.first_walk_duration_ * 60;
      s1.arrival_.schedule_timestamp_ = s1.arrival_.timestamp_;
      s1.arrival_.timestamp_reason_ =
          j.stops_[p.from_].departure_.timestamp_reason_;
      s1.departure_ = s1.arrival_;

      auto& s2 = j.stops_.at(p.from_ + 2);
      s2.eva_no_ = s.to_station_id_;
      s2.name_ = s.to_station_name_;
      s2.lat_ = s.to_station_pos_.lat_;
      s2.lng_ = s.to_station_pos_.lng_;
      s2.arrival_.valid_ = true;
      s2.arrival_.timestamp_ = s1.departure_.timestamp_ + s.bike_duration_ * 60;
      s2.arrival_.schedule_timestamp_ = s2.arrival_.timestamp_;
      s2.arrival_.timestamp_reason_ =
          j.stops_[p.from_ + 1].departure_.timestamp_reason_;
      s2.departure_ = s2.arrival_;

      get_transport(j, p.from_, p.from_ + 1).mumo_type_ =
          to_string(mumo_type::FOOT);
      get_transport(j, p.from_ + 1, p.from_ + 2).mumo_type_ =
          p.e_->gbfs_->vehicle_type_;
      get_transport(j, p.from_ + 2, p.from_ + 3).mumo_type_ =
          to_string(mumo_type::FOOT);
    }

    // free bike:
    // replace: X --walk[type:gbfs]--> P
    // to: X --walk--> (B) --bike--> P
    // replace: P -->walk[type:gbfs]--> X
    // to: P --walk--> (B) --bike--> X
    else if (std::holds_alternative<gbfs_edge::free_bike>(p.e_->gbfs_->bike_)) {
      auto const& b = std::get<gbfs_edge::free_bike>(p.e_->gbfs_->bike_);

      auto& t = get_transport(j, p.from_, p.to_);
      auto str = split_transport(j, patches, t);

      str.parking_stop_.eva_no_ = b.id_;
      str.parking_stop_.name_ = b.id_;
      str.parking_stop_.lat_ = b.pos_.lat_;
      str.parking_stop_.lng_ = b.pos_.lng_;
      str.parking_stop_.arrival_.valid_ = true;
      str.parking_stop_.arrival_.timestamp_ =
          j.stops_[p.from_].departure_.timestamp_ + b.walk_duration_ * 60;
      str.parking_stop_.arrival_.schedule_timestamp_ =
          str.parking_stop_.arrival_.timestamp_;
      str.parking_stop_.arrival_.timestamp_reason_ =
          j.stops_[p.from_].departure_.timestamp_reason_;
      str.parking_stop_.departure_ = str.parking_stop_.arrival_;

      str.first_transport_.mumo_type_ = to_string(mumo_type::FOOT);
      str.second_transport_.mumo_type_ = to_string(mumo_type::BIKE);
    }
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
  auto routing_response =
      response_msg ? motis_content(RoutingResponse, response_msg) : nullptr;
  auto journeys = routing_response == nullptr
                      ? std::vector<journey>{}
                      : message_to_journeys(routing_response);

  MOTIS_START_TIMING(direct_connection_timing);
  auto const direct =
      get_direct_connections(q_start, q_dest, req, profiles, edge_mapping);
  stats.dominated_by_direct_connection_ =
      remove_dominated_journeys(journeys, direct);
  add_direct_connections(journeys, direct, q_start, q_dest, req);
  MOTIS_STOP_TIMING(direct_connection_timing);
  stats.direct_connection_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(direct_connection_timing));

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

    auto gbfs_patches = std::vector<parking_patch>{};
    auto parking_patches = std::vector<parking_patch>{};
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
        parking_patches.emplace_back(e, t.from_, t.to_);
      } else if (e->type_ == mumo_type::GBFS) {
        gbfs_patches.emplace_back(e, t.from_, t.to_);
      }
    }
    apply_parking_patches(journey, parking_patches);
    apply_gbfs_patches(journey, gbfs_patches);
  }

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
  auto all_stats =
      routing_response == nullptr
          ? std::vector<Offset<Statistics>>{}
          : utl::to_vec(*routing_response->statistics(),
                        [&](Statistics const* stats) {
                          return motis_copy_table(Statistics, mc, stats);
                        });
  for (auto const& s : mumo_stats) {
    all_stats.emplace_back(to_fbs(mc, s));
  }
  all_stats.emplace_back(to_fbs(mc, to_stats_category("intermodal", stats)));

  auto interval_begin = uint64_t{};
  auto interval_end = uint64_t{};
  if (routing_response != nullptr) {
    interval_begin = routing_response->interval_begin();
    interval_end = routing_response->interval_end();
  } else {
    switch (req->start_type()) {
      case IntermodalStart_PretripStart:
        interval_begin =
            reinterpret_cast<IntermodalPretripStart const*>(req->start())
                ->interval()
                ->begin();
        interval_end =
            reinterpret_cast<IntermodalPretripStart const*>(req->start())
                ->interval()
                ->begin();
        break;

      case IntermodalStart_IntermodalPretripStart:
        interval_begin = reinterpret_cast<PretripStart const*>(req->start())
                             ->interval()
                             ->begin();
        interval_end = reinterpret_cast<PretripStart const*>(req->start())
                           ->interval()
                           ->begin();
        break;

      case IntermodalStart_IntermodalOntripStart:
        interval_begin = interval_end =
            reinterpret_cast<IntermodalOntripStart const*>(req->start())
                ->departure_time();
        break;

      case IntermodalStart_OntripStationStart:
        interval_begin = interval_end =
            reinterpret_cast<OntripStationStart const*>(req->start())
                ->departure_time();
        break;

      case IntermodalStart_OntripTrainStart:
        interval_begin = interval_end =
            reinterpret_cast<OntripTrainStart const*>(req->start())
                ->arrival_time();
        break;

      case IntermodalStart_NONE: break;
    }
  }

  mc.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          mc, mc.CreateVectorOfSortedTables(&all_stats),
          mc.CreateVector(connections), interval_begin, interval_end,
          mc.CreateVector(utl::to_vec(
              direct,
              [&mc](direct_connection const& c) { return to_fbs(mc, c); })))
          .Union());
  return make_msg(mc);
}

msg_ptr intermodal::route(msg_ptr const& msg) {
  auto const req = motis_content(IntermodalRoutingRequest, msg);
  message_creator mc;
  statistics stats{};

  auto span = motis_tracer->StartSpan("intermodal::route");
  auto scope = opentelemetry::trace::Scope{span};

  MOTIS_START_TIMING(total_timing);

  auto const start = parse_query_start(mc, req);
  auto const dest = parse_query_dest(mc, req);

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
    std::lock_guard const guard(mumo_stats_mutex);
    mumo_stats.emplace_back(std::move(s));
  };

  std::vector<ctx::future_ptr<ctx_data, void>> futures;

  using namespace std::placeholders;
  if (req->search_dir() == SearchDir_Forward) {
    metrics_->fwd_requests_.Increment();
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_, dest.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      STATION_START, _1, start.pos_, _2, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_, *metrics_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_, start.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             _1, STATION_END, _2, dest.pos_, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_, *metrics_);
      }));
    }
  } else {
    metrics_->bwd_requests_.Increment();
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_, dest.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      _1, STATION_START, _2, start.pos_, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_, *metrics_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_, start.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             STATION_END, _1, dest.pos_, _2, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_, *metrics_);
      }));
    }
  }

  ctx::await_all(futures);
  MOTIS_STOP_TIMING(mumo_edge_timing);
  metrics_->mumo_edges_time_.Observe(MOTIS_TIMING_S(mumo_edge_timing));

  stats.start_edges_ = deps.size();
  stats.destination_edges_ = arrs.size();
  stats.mumo_edge_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(mumo_edge_timing));

  std::vector<mumo_edge const*> edge_mapping;
  auto edges = write_edges(mc, deps, arrs, edge_mapping);

  auto routing_resp = msg_ptr{};
  if ((!start.is_intermodal_ || !deps.empty()) &&
      (!dest.is_intermodal_ || !arrs.empty())) {
    auto const router =
        (req->router() == nullptr || req->router()->Length() == 0U)
            ? ((req->search_type() == SearchType_Default ||
                req->search_type() == SearchType_Accessibility) &&
               start.start_type_ != Start_OntripTrainStart)
                  ? router_
                  : "/routing"
            : req->router()->str();

    auto const via = mc.CreateVector(
        req->via() != nullptr
            ? utl::to_vec(*req->via(),
                          [&mc](Via const* via) {
                            return motis_copy_table(Via, mc, via);
                          })
            : std::vector<Offset<Via>>{});

    mc.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(
            mc, start.start_type_, start.start_, dest.station_,
            req->search_type(), req->search_dir(), via, mc.CreateVector(edges),
            true, true, !start.is_intermodal_, 0, timeout_,
            req->allowed_claszes() == nullptr
                ? 0
                : mc.CreateVector(req->allowed_claszes()->Data(),
                                  req->allowed_claszes()->size()),
            req->max_transfers(), req->bike_transport(),
            req->min_transfer_time(), req->transfer_time_factor())
            .Union(),
        router);

    MOTIS_START_TIMING(routing_timing);
    routing_resp = motis_call(make_msg(mc))->val();
    MOTIS_STOP_TIMING(routing_timing);

    stats.routing_duration_ =
        static_cast<uint64_t>(MOTIS_TIMING_MS(routing_timing));
  }

  auto const response =
      postprocess_response(routing_resp, start, dest, req, edge_mapping, stats,
                           revise_, mumo_stats, ppr_profiles_);

  MOTIS_STOP_TIMING(total_timing);
  metrics_->total_time_.Observe(MOTIS_TIMING_S(total_timing));

  return response;
}

}  // namespace motis::intermodal
