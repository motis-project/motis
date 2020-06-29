#include "motis/railviz/railviz.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/core/access/bfs.h"
#include "motis/core/access/edge_access.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/transport_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"

#include "motis/path/path_data.h"

#include "motis/railviz/train_retriever.h"
#include "motis/railviz/trains_response_builder.h"

using namespace motis::access;
using namespace motis::logging;
using namespace motis::module;
using namespace flatbuffers;

constexpr auto const MAX_ZOOM = 20U;

namespace motis::railviz {

railviz::railviz() : module("RailViz", "railviz") {
  param(initial_permalink_, "initial_permalink",
        "prefix of: /lat/lng/zoom/bearing/pitch/timestamp");
  param(tiles_redirect_, "tiles_redirect",
        "http://url.to/tiles/server (empty: use this MOTIS instance)");
}

railviz::~railviz() = default;

mcd::hash_map<std::pair<int, int>, geo::box> bounding_boxes(schedule const& s) {
  mcd::hash_map<std::pair<int, int>, geo::box> boxes;

  auto const from_fbs = [](path::Box const* b) {
    return geo::make_box(
        {geo::latlng{b->south_west()->lat(), b->south_west()->lng()},
         geo::latlng{b->north_east()->lat(), b->north_east()->lng()}});
  };

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
          from_fbs(box);
    }
  } catch (std::system_error const& e) {
    LOG(logging::warn) << "bounding box request failed: " << e.what();
  }

  return boxes;
}

// est. rationale: cut off 20% outliers each and assume 2x2 tiles are on screen
std::string estimate_initial_permalink(schedule const& sched) {
  auto const get_quantiles = [](std::vector<double> coords) {
    utl::erase_if(coords, [](auto const c) { return c == 0.; });
    if (coords.empty()) {
      return std::make_pair(0., 0.);
    }
    if (coords.size() < 10) {
      return std::make_pair(coords.front(), coords.back());
    }

    std::sort(begin(coords), end(coords));
    constexpr auto const kQuantile = .8;
    return std::make_pair(coords.at(coords.size() * (1 - kQuantile)),
                          coords.at(coords.size() * (kQuantile)));
  };

  auto const [lat_min, lat_max] = get_quantiles(
      utl::to_vec(sched.stations_, [](auto const& s) { return s->lat(); }));
  auto const [lng_min, lng_max] = get_quantiles(
      utl::to_vec(sched.stations_, [](auto const& s) { return s->lng(); }));

  auto const fixed0 = tiles::latlng_to_fixed({lat_min, lng_min});
  auto const fixed1 = tiles::latlng_to_fixed({lat_max, lng_max});

  auto const center = tiles::fixed_to_latlng(
      {(fixed0.x() + fixed1.x()) / 2, (fixed0.y() + fixed1.y()) / 2});

  auto const d = std::max(std::abs(fixed0.x() - fixed1.x()),
                          std::abs(fixed0.y() - fixed1.y()));

  auto zoom = int{0};
  for (; zoom < (tiles::kMaxZoomLevel - 1); ++zoom) {
    if (((tiles::kTileSize * 2ULL) *
         (1ULL << (tiles::kMaxZoomLevel - (zoom + 1)))) < d) {
      break;
    }
  }

  return fmt::format("/{:.7}/{:.7}/{}", center.lat_, center.lng_, zoom);
}

void railviz::init(motis::module::registry& reg) {
  reg.register_op("/railviz/map_config",
                  [this](auto const& msg) { return get_map_config(msg); });
  reg.register_op("/railviz/get_trip_guesses", &railviz::get_trip_guesses);
  reg.register_op("/railviz/get_station", &railviz::get_station);
  reg.register_op("/railviz/get_trains",
                  [this](msg_ptr const& msg) { return get_trains(msg); });
  reg.register_op("/railviz/get_trips",
                  [this](msg_ptr const& msg) { return get_trips(msg); });

  reg.subscribe("/init", [this]() {
    auto const& s = get_sched();
    train_retriever_ = std::make_unique<train_retriever>(s, bounding_boxes(s));

    if (initial_permalink_.empty()) {
      initial_permalink_ = estimate_initial_permalink(s);
      LOG(logging::info) << "est. initial_permalink: " << initial_permalink_;
    }
  });

  reg.subscribe("/rt/update", [this](msg_ptr const& msg) {
    using rt::RtUpdates;
    if (train_retriever_) {
      train_retriever_->update(motis_content(RtUpdates, msg));
    }
    return nullptr;
  });
}

msg_ptr railviz::get_map_config(msg_ptr const&) {
  message_creator mc;
  mc.create_and_finish(
      MsgContent_RailVizMapConfigResponse,
      CreateRailVizMapConfigResponse(mc, mc.CreateString(initial_permalink_),
                                     mc.CreateString(tiles_redirect_))
          .Union());
  return make_msg(mc);
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

msg_ptr railviz::get_trains(msg_ptr const& msg) const {
  logging::scoped_timer timer("get_trains");

  auto const req = motis_content(RailVizTrainsRequest, msg);
  auto const& sched = get_schedule();

  auto const start_time = unix_to_motistime(sched, req->start_time());
  auto const end_time = unix_to_motistime(sched, req->end_time());

  trains_response_builder trb{
      sched, find_shared_data<path::path_data>(path::PATH_DATA_KEY),
      req->zoom_geo()};
  for (auto const& ev : train_retriever_->trains(
           start_time, end_time, req->max_trains(), req->last_trains(),
           geo::make_box(
               {geo::latlng{req->corner1()->lat(), req->corner1()->lng()},
                geo::latlng{req->corner2()->lat(), req->corner2()->lng()}}),
           req->zoom_bounds())) {
    trb.add_train(ev);
  }
  return trb.finish();
}

msg_ptr railviz::get_trips(msg_ptr const& msg) {
  auto const* req = motis_content(RailVizTripsRequest, msg);
  auto const& sched = get_schedule();

  trains_response_builder trb{
      sched, find_shared_data<path::path_data>(path::PATH_DATA_KEY), MAX_ZOOM};
  for (auto const* fbs_trp : *req->trips()) {
    auto const trp = from_fbs(sched, fbs_trp);
    auto const k = ev_key{trp->edges_->at(0), trp->lcon_idx_, event_type::DEP};
    trb.add_train_full(k);
  }
  return trb.finish();
}

}  // namespace motis::railviz
