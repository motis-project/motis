#include "motis/railviz/railviz.h"

#include "utl/concat.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/core/access/bfs.h"
#include "motis/core/access/edge_access.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/transport_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"

#include "motis/railviz/path_resolver.h"
#include "motis/railviz/train_retriever.h"

using namespace motis::logging;
using namespace motis::module;
using namespace flatbuffers;

constexpr auto const MAX_ZOOM = 18U;

namespace motis::railviz {

railviz::railviz() : module("RailViz", "railviz") {}

railviz::~railviz() = default;

mcd::hash_map<std::pair<int, int>, geo::box> bounding_boxes(schedule const& s) {
  mcd::hash_map<std::pair<int, int>, geo::box> boxes;

  try {
    using path::PathBoxesResponse;
    auto const res = motis_call(make_no_msg("/path/boxes"))->val();
    for (auto const& box : *motis_content(PathBoxesResponse, res)->boxes()) {
      auto const it_a = s.eva_to_station_.find(box->station_id_a()->str());
      auto const it_b = s.eva_to_station_.find(box->station_id_b()->str());
      if (it_a == end(s.eva_to_station_) || it_b == end(s.eva_to_station_)) {
        continue;
      }
      boxes[{std::min(it_a->second->index_, it_b->second->index_),
             std::max(it_a->second->index_, it_b->second->index_)}] =
          geo::from_fbs(box);
    }
  } catch (std::system_error const& e) {
    LOG(logging::warn) << "bounding box request failed: " << e.what();
  }

  return boxes;
}

void railviz::init(motis::module::registry& reg) {
  reg.register_op("/railviz/get_trip_guesses", &railviz::get_trip_guesses);
  reg.register_op("/railviz/get_station", &railviz::get_station);
  reg.register_op("/railviz/get_trains",
                  [this](msg_ptr const& msg) { return get_trains(msg); });
  reg.register_op("/railviz/get_trips", &railviz::get_trips);

  reg.subscribe("/init", [this]() {
    auto const& s = synced_sched<RO>().sched();
    train_retriever_ = std::make_unique<train_retriever>(s, bounding_boxes(s));
  });
}

msg_ptr railviz::get_trip_guesses(msg_ptr const& msg) {
  auto const req = motis_content(RailVizTripGuessRequest, msg);

  auto const& sched = get_schedule();
  auto it =
      std::lower_bound(begin(sched.trips_), end(sched.trips_),
                       std::make_pair(primary_trip_id{0U, req->train_num(), 0U},
                                      static_cast<trip*>(nullptr)));

  auto const cmp = [&](trip const* a, trip const* b) {
    return std::abs(static_cast<int64_t>(a->id_.primary_.time_)) <
           std::abs(static_cast<int64_t>(b->id_.primary_.time_));
  };

  std::vector<trip const*> trips;
  while (it != end(sched.trips_) && it->first.train_nr_ == req->train_num()) {
    trips.emplace_back(it->second);

    auto const interesting_size =
        std::min(req->guess_count(), static_cast<unsigned>(trips.size()));
    std::nth_element(begin(trips), std::next(begin(trips), interesting_size),
                     end(trips), cmp);
    trips.resize(interesting_size);

    ++it;
  }

  std::sort(begin(trips), end(trips), cmp);

  auto const get_first_dep_ci = [&](trip const* trp) {
    auto const& lcon =
        trp->edges_->front()->m_.route_edge_.conns_[trp->lcon_idx_];
    auto const& merged = *sched.merged_trips_[lcon.trips_];
    auto const it = std::find(begin(merged), end(merged), trp);
    utl::verify(it != end(merged), "trip not found in trip");
    auto const merged_ci_idx = std::distance(begin(merged), it);

    auto i = 0u;
    for (auto ci = lcon.full_con_->con_info_; ci != nullptr;
         ci = ci->merged_with_) {
      if (i == merged_ci_idx) {
        return ci;
      }
      ++i;
    }

    throw utl::fail("merged with ci not found");
  };

  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RailVizTripGuessResponse,
      CreateRailVizTripGuessResponse(
          fbb, fbb.CreateVector(utl::to_vec(
                   trips,
                   [&](trip const* trp) {
                     return CreateTrip(
                         fbb,
                         to_fbs(fbb, *sched.stations_.at(
                                         trp->id_.primary_.station_id_)),
                         CreateTripInfo(
                             fbb, to_fbs(sched, fbb, trp),
                             to_fbs(sched, fbb, get_first_dep_ci(trp), trp)));
                   })))
          .Union());
  return make_msg(fbb);
}

msg_ptr railviz::get_station(msg_ptr const& msg) {
  auto const req = motis_content(RailVizStationRequest, msg);

  auto const& sched = get_schedule();
  auto const t = unix_to_motistime(sched, req->time());
  auto const station = get_station_node(sched, req->station_id()->str());

  struct ev_info {
    motis::time t_{0};
    ev_key k_;
  };

  auto const make_ev_info = [&](ev_key const& k) {
    return ev_info{
        req->by_schedule_time() ? get_schedule_time(sched, k) : k.get_time(),
        k};
  };

  auto const is_filtered = [&](ev_info const& ev) {
    return (req->direction() == Direction_LATER && ev.t_ <= t) ||
           (req->direction() == Direction_EARLIER && ev.t_ >= t);
  };

  auto const cmp = [&](ev_info const& a, ev_info const& b) {
    return std::abs(a.t_ - t) < std::abs(b.t_ - t);
  };

  // get top $event_count events
  // matching the direction (earlier/later)
  std::vector<ev_info> events;
  auto const on_ev = [&](ev_key const& k) {
    auto const ev = make_ev_info(k);
    if (is_filtered(ev)) {
      return;
    }

    events.push_back(ev);

    auto const interesting_size =
        std::min(req->event_count(), static_cast<unsigned>(events.size()));
    std::nth_element(begin(events), std::next(begin(events), interesting_size),
                     end(events), cmp);
    events.resize(interesting_size);
  };

  // collect departure events (only in allowed)
  for (auto const& se : station->edges_) {
    if (se.type() == edge::INVALID_EDGE || !se.to_->is_route_node()) {
      continue;
    }

    for (auto const& re : se.to_->edges_) {
      if (re.empty()) {
        continue;
      }

      for (auto i = 0U; i < re.m_.route_edge_.conns_.size(); ++i) {
        on_ev(ev_key{&re, i, event_type::DEP});
      }
    }
  }

  // collect arrival events (only out allowed)
  for (auto const& se : station->incoming_edges_) {
    if (se->type() == edge::INVALID_EDGE || !se->from_->is_route_node()) {
      continue;
    }

    for (auto const& re : se->from_->incoming_edges_) {
      if (re->empty()) {
        continue;
      }

      for (auto i = 0U; i < re->m_.route_edge_.conns_.size(); ++i) {
        on_ev(ev_key{trip::route_edge{re}, i, event_type::ARR});
      }
    }
  }

  std::sort(begin(events), end(events),
            [](ev_info const& a, ev_info const& b) { return a.t_ < b.t_; });

  // convert to message buffer
  message_creator fbb;

  auto const get_trips = [&](ev_key const& k) {
    std::vector<Offset<TripInfo>> trips;

    auto const& merged_trips = *sched.merged_trips_[k.lcon()->trips_];
    auto merged_trips_idx = 0u;
    for (auto ci = k.lcon()->full_con_->con_info_; ci != nullptr;
         ci = ci->merged_with_, ++merged_trips_idx) {
      auto const& trp = merged_trips.at(merged_trips_idx);
      trips.push_back(CreateTripInfo(fbb, to_fbs(sched, fbb, trp),
                                     to_fbs(sched, fbb, ci, trp)));
    }

    return fbb.CreateVector(trips);
  };

  auto const get_track = [&](ev_info const& ev) {
    auto const fcon = ev.k_.lcon()->full_con_;
    auto const track = ev.k_.is_arrival() ? fcon->a_track_ : fcon->d_track_;
    return fbb.CreateString(sched.tracks_[track]);
  };

  fbb.create_and_finish(
      MsgContent_RailVizStationResponse,
      CreateRailVizStationResponse(
          fbb, to_fbs(fbb, *sched.stations_.at(station->id_)),
          fbb.CreateVector(utl::to_vec(
              events,
              [&](ev_info const& ev) {
                auto const di = get_delay_info(sched, ev.k_);
                return CreateEvent(
                    fbb, get_trips(ev.k_), to_fbs(ev.k_.ev_type_),
                    CreateEventInfo(
                        fbb, motis_to_unixtime(sched, ev.k_.get_time()),
                        motis_to_unixtime(sched, di.get_schedule_time()),
                        get_track(ev),
                        fbb.CreateString(
                            sched.tracks_[get_schedule_track(sched, ev.k_)]),
                        ev.k_.lcon()->valid_ != 0u, to_fbs(di.get_reason())));
              })))
          .Union());

  return make_msg(fbb);
}

std::vector<Offset<Train>> events_to_trains(
    schedule const& sched, FlatBufferBuilder& fbb, path_resolver& pr,
    std::vector<ev_key> const& evs,  //
    std::map<int, int>& routes, std::vector<Offset<Route>>& fbs_routes,
    std::vector<std::set<trip::route_edge>>& route_edges) {
  auto const get_route_segments = [&](std::set<trip::route_edge> const& edges) {
    return fbb.CreateVector(utl::to_vec(edges, [&](trip::route_edge const& e) {
      auto const& from = *sched.stations_[e->from_->get_station()->id_];
      auto const& to = *sched.stations_[e->to_->get_station()->id_];
      return CreateSegment(
          fbb, fbb.CreateString(from.eva_nr_), fbb.CreateString(to.eva_nr_),
          CreatePolyline(fbb, fbb.CreateVector(pr.get_segment_path(e))));
    }));
  };

  auto const get_trips = [&sched, &fbb](ev_key const& k) {
    return fbb.CreateVector(
        utl::to_vec(*sched.merged_trips_[k.lcon()->trips_],
                    [&](trip const* trp) { return to_fbs(sched, fbb, trp); }));
  };

  auto const get_route = [&fbb, &routes, &fbs_routes, &route_edges,
                          &get_route_segments](ev_key const& k) -> int {
    auto const insert = routes.emplace(k.get_node()->route_,
                                       static_cast<int>(fbs_routes.size()));
    if (insert.second) {
      auto const edges = route_bfs(k, bfs_direction::BOTH, false);
      route_edges.emplace_back(edges);
      fbs_routes.emplace_back(CreateRoute(fbb, get_route_segments(edges)));
    }
    return insert.first->second;
  };

  auto const service_names = [&fbb, &sched](ev_key const& k) {
    std::vector<std::string> names;
    auto c_info = k.lcon()->full_con_->con_info_;
    while (c_info != nullptr) {
      names.push_back(get_service_name(sched, c_info));
      c_info = c_info->merged_with_;
    }
    return fbb.CreateVector(utl::to_vec(names, [&](std::string const& name) {
      return fbb.CreateString(name);
    }));
  };

  return utl::to_vec(evs, [&](ev_key const& dep) {
    auto const route = get_route(dep);
    auto const& edges = route_edges[route];
    auto const segment_idx =
        std::distance(begin(edges), edges.find(dep.route_edge_));
    auto const arr = dep.get_opposite();

    auto const dep_di = get_delay_info(sched, dep);
    auto const arr_di = get_delay_info(sched, arr);

    return CreateTrain(fbb, service_names(dep), dep.lcon()->full_con_->clasz_,
                       motis_to_unixtime(sched, dep.get_time()),
                       motis_to_unixtime(sched, arr.get_time()),
                       motis_to_unixtime(sched, dep_di.get_schedule_time()),
                       motis_to_unixtime(sched, arr_di.get_schedule_time()),
                       to_fbs(dep_di.get_reason()), to_fbs(arr_di.get_reason()),
                       route, segment_idx, get_trips(dep));
  });
}

std::vector<Offset<Station>> get_stations(
    schedule const& sched, FlatBufferBuilder& fbb,
    std::vector<std::set<trip::route_edge>>& route_edges) {
  std::set<int> stations_indices;
  for (auto const& route : route_edges) {
    for (auto const& e : route) {
      stations_indices.emplace(e->from_->get_station()->id_);
      stations_indices.emplace(e->to_->get_station()->id_);
    }
  }

  return utl::to_vec(stations_indices, [&sched, &fbb](int const station_idx) {
    auto const& station = *sched.stations_[station_idx];
    auto const pos = Position(station.width_, station.length_);
    return CreateStation(fbb, fbb.CreateString(station.eva_nr_),
                         fbb.CreateString(station.name_), &pos);
  });
};

msg_ptr railviz::get_trains(msg_ptr const& msg) const {
  logging::scoped_timer timer("get_trains");

  auto const req = motis_content(RailVizTrainsRequest, msg);
  auto const& sched = get_schedule();

  message_creator fbb;

  std::map<int, int> routes;
  std::vector<Offset<Route>> fbs_routes;
  std::vector<std::set<trip::route_edge>> route_edges;
  path_resolver pr(sched, req->zoom_level());

  auto const fbs_trains = events_to_trains(
      sched, fbb, pr,
      train_retriever_->trains(
          unix_to_motistime(sched, req->start_time()),
          unix_to_motistime(sched, req->end_time()), req->max_trains(),
          {{req->corner1()->lat(), req->corner1()->lng()},
           {req->corner2()->lat(), req->corner2()->lng()}}),
      routes, fbs_routes, route_edges);

  fbb.create_and_finish(
      MsgContent_RailVizTrainsResponse,
      CreateRailVizTrainsResponse(
          fbb, fbb.CreateVector(fbs_trains), fbb.CreateVector(fbs_routes),
          fbb.CreateVector(get_stations(sched, fbb, route_edges)))
          .Union());

  LOG(info) << "path module requests: " << pr.get_req_count();

  return make_msg(fbb);
}

msg_ptr railviz::get_trips(msg_ptr const& msg) {
  auto const& sched = get_schedule();
  message_creator fbb;

  std::map<int, int> routes;
  std::vector<Offset<Route>> fbs_routes;
  std::vector<std::set<trip::route_edge>> route_edges;
  path_resolver pr(sched, MAX_ZOOM);

  std::vector<Offset<Train>> fbs_trains;
  for (auto const& trip : *motis_content(RailVizTripsRequest, msg)->trips()) {
    auto const trp = from_fbs(sched, trip);
    auto const first_dep =
        ev_key{trp->edges_->at(0), trp->lcon_idx_, event_type::DEP};
    utl::concat(
        fbs_trains,
        events_to_trains(
            sched, fbb, pr,
            utl::to_vec(route_bfs(first_dep, bfs_direction::BOTH, false),
                        [&trp](trip::route_edge const& e) {
                          return ev_key{e, trp->lcon_idx_, event_type::DEP};
                        }),
            routes, fbs_routes, route_edges));
  }

  fbb.create_and_finish(
      MsgContent_RailVizTrainsResponse,
      CreateRailVizTrainsResponse(
          fbb, fbb.CreateVector(fbs_trains), fbb.CreateVector(fbs_routes),
          fbb.CreateVector(get_stations(sched, fbb, route_edges)))
          .Union());

  return make_msg(fbb);
}

}  // namespace motis::railviz
