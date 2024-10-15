#include "motis/raptor/get_raptor_timetable.h"

#include <thread>
#include <tuple>

#include "utl/concat.h"
#include "utl/erase_duplicates.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

namespace motis::raptor {

namespace log = motis::logging;

struct transformable_footpath {
  transformable_footpath(stop_id const from, stop_id const to,
                         time const duration)
      : from_{from}, to_{to}, duration_{duration} {}

  stop_id from_;
  stop_id to_;
  time duration_;
};

struct transformable_stop {
  std::vector<transformable_footpath> footpaths_;
  std::vector<transformable_footpath> incoming_footpaths_;
  std::vector<route_id> stop_routes_;
  std::vector<stop_id> equivalent_;
  time transfer_time_{invalid<time>};
  std::string eva_{};
  unsigned motis_station_index_{invalid<unsigned>};
};

struct raptor_lcon {
  raptor_lcon(stop_id const from, stop_id const to, time const dep,
              time const arrival, bool const in_allowed, bool const out_allowed,
              light_connection const* lc)
      : from_{from},
        to_{to},
        departure_{dep},
        arrival_{arrival},
        in_allowed_{in_allowed},
        out_allowed_{out_allowed},
        lcon_{lc} {}
  stop_id from_;
  stop_id to_;
  time departure_;
  time arrival_;
  bool in_allowed_;
  bool out_allowed_;
  light_connection const* lcon_;
};

struct transformable_trip {
  std::vector<raptor_lcon> lcons_;
  std::vector<stop_time> stop_times_;
  std::vector<stop_attributes> stop_attr_;
  std::string dbg_;
};

struct transformable_route {
  std::vector<transformable_trip> trips_;
  std::vector<stop_id> route_stops_;
};

struct transformable_timetable {
  std::vector<transformable_stop> stations_;
  std::vector<transformable_route> routes_;
};

std::vector<stop_time> get_stop_times_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  std::vector<stop_time> stop_times(lcons.size() + 1);
  for (auto idx = 0; idx < lcons.size(); ++idx) {
    auto const& lcon = lcons[idx];
    if (lcon.in_allowed_) {
      stop_times[idx].departure_ = lcon.departure_;
    }
    if (lcon.out_allowed_) {
      stop_times[idx + 1].arrival_ = lcon.arrival_;
    }
  }
  return stop_times;
}

std::vector<stop_attributes> get_stop_attributes_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  auto const edge_count = lcons.size();
  auto const stop_count = edge_count + 1;

  std::vector<stop_attributes> stop_attributes(stop_count);

  // first stop has no inbound edge; therefore also no occupancy
  stop_attributes[0].inbound_occupancy_ = 0;

  for (auto idx = 1; idx < stop_count; ++idx) {
    stop_attributes[idx].inbound_occupancy_ = lcons[idx - 1].lcon_->occupancy_;
  }

  return stop_attributes;
}

std::vector<stop_id> get_route_stops_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  std::vector<stop_id> route_stops;
  route_stops.reserve(lcons.size() + 1);
  for (auto const& lcon : lcons) {
    route_stops.push_back(lcon.from_);
  }
  route_stops.push_back(lcons.back().to_);
  return route_stops;
}

void init_stops(schedule const& sched, std::vector<transformable_stop>& tss) {
  for (auto s_id = 0; s_id < sched.stations_.size(); ++s_id) {
    auto& s = tss[s_id];
    auto const& s_ptr = sched.stations_[s_id];

    s.motis_station_index_ = s_ptr->index_;
    s.transfer_time_ = s_ptr->transfer_time_;
    s.eva_ = s_ptr->eva_nr_;

    s.equivalent_.push_back(s_id);
    for (auto const& equi_ptr : s_ptr->equivalent_) {
      if (equi_ptr->index_ == s_id) {
        continue;
      }
      s.equivalent_.push_back(equi_ptr->index_);
    }
  }
}

void init_stop_routes(transformable_timetable& ttt) {
  for (auto r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    auto const& route = ttt.routes_[r_id];
    auto const& trip = route.trips_.front();

    for (auto const& lcon : trip.lcons_) {
      ttt.stations_[lcon.from_].stop_routes_.push_back(r_id);
    }

    ttt.stations_[trip.lcons_.back().to_].stop_routes_.push_back(r_id);
  }

  for (auto& station : ttt.stations_) {
    utl::erase_duplicates(station.stop_routes_);
  }
}

void init_routes(schedule const& sched, std::vector<transformable_route>& rs) {
  using namespace motis::access;

  rs.resize(sched.expanded_trips_.index_size() - 1);

  route_id r_id = 0;
  for (auto const route_trips : sched.expanded_trips_) {
    auto& t_route = rs[r_id];

    t_route.trips_.resize(route_trips.size());

    auto const& first_trip = route_trips[0];
    auto const in_allowed =
        utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
          return ts.get_route_node()->is_in_allowed();
        });

    auto const out_allowed =
        utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
          return ts.get_route_node()->is_out_allowed();
        });

    trip_id t_id = 0;
    for (auto const& trip : route_trips) {
      auto& t_trip = t_route.trips_[t_id];

      for (auto const section : sections(trip)) {
        auto const& lc = section.lcon();
        auto const from = section.from_station_id();
        auto const to = section.to_station_id();
        auto const from_in_allowed = in_allowed[section.index()];
        auto const to_out_allowed = out_allowed[section.index() + 1];

        t_trip.lcons_.emplace_back(from, to, lc.d_time_, lc.a_time_,
                                   from_in_allowed, to_out_allowed, &lc);
      }

      t_trip.stop_times_ = get_stop_times_from_lcons(t_trip.lcons_);
      t_trip.stop_attr_ = get_stop_attributes_from_lcons(t_trip.lcons_);
      t_trip.dbg_ = std::string{trip->dbg_.str()};

      ++t_id;
    }

    t_route.route_stops_ =
        get_route_stops_from_lcons(t_route.trips_.front().lcons_);

    ++r_id;
  }
}

void add_footpaths(schedule const& sched, std::vector<transformable_stop>& s) {
  for (auto s_id = 0; s_id < s.size(); ++s_id) {
    auto const& motis_station = sched.stations_[s_id];
    auto& raptor_station = s[s_id];

    auto const& motis_fp_to_transformable = [](motis::footpath const& f) {
      return transformable_footpath(f.from_station_, f.to_station_,
                                    f.duration_);
    };

    std::transform(std::begin(motis_station->outgoing_footpaths_),
                   std::end(motis_station->outgoing_footpaths_),
                   std::back_inserter(raptor_station.footpaths_),
                   motis_fp_to_transformable);

    std::transform(std::begin(motis_station->incoming_footpaths_),
                   std::end(motis_station->incoming_footpaths_),
                   std::back_inserter(raptor_station.incoming_footpaths_),
                   motis_fp_to_transformable);
  }
}

std::unique_ptr<raptor_timetable> create_raptor_timetable(
    transformable_timetable const& ttt) {
  auto tt = std::make_unique<raptor_timetable>();

  tt->stops_.reserve(ttt.stations_.size() + 1);
  tt->transfer_times_.reserve(ttt.stations_.size());
  tt->incoming_footpaths_.resize(ttt.stations_.size());

  for (auto s_id = 0U; s_id < ttt.stations_.size(); ++s_id) {
    auto const& t_stop = ttt.stations_[s_id];

    auto footpaths_idx = static_cast<footpaths_index>(tt->footpaths_.size());
    auto sr_idx = static_cast<stop_routes_index>(tt->stop_routes_.size());
    auto fc = static_cast<footpath_count>(t_stop.footpaths_.size());
    auto rc = static_cast<route_count>(t_stop.stop_routes_.size());
    tt->stops_.emplace_back(fc, rc, footpaths_idx, sr_idx);

    utl::concat(tt->stop_routes_, t_stop.stop_routes_);

    for (auto const& f : t_stop.footpaths_) {
      auto const transfer_time = ttt.stations_[f.from_].transfer_time_;
      tt->footpaths_.emplace_back(f.to_, f.duration_ - transfer_time);
    }

    for (auto const& f : t_stop.incoming_footpaths_) {
      auto const transfer_time = ttt.stations_[f.from_].transfer_time_;
      tt->incoming_footpaths_[s_id].emplace_back(
          f.from_, f.duration_ - transfer_time);
    }

    auto const transfer_time = ttt.stations_[s_id].transfer_time_;
    tt->transfer_times_.push_back(transfer_time);
  }

  auto footpaths_idx = static_cast<footpaths_index>(tt->footpaths_.size());
  auto sr_idx = static_cast<stop_routes_index>(tt->stop_routes_.size());
  tt->stops_.emplace_back(0, 0, footpaths_idx, sr_idx);

  tt->routes_.reserve(ttt.routes_.size() + 1);
  for (auto const& t_route : ttt.routes_) {
    auto sc = static_cast<stop_count>(t_route.route_stops_.size());
    auto tc = static_cast<trip_count>(t_route.trips_.size());
    auto stop_times_idx = static_cast<stop_times_index>(tt->stop_times_.size());
    auto rs_idx = static_cast<route_stops_index>(tt->route_stops_.size());

    tt->routes_.emplace_back(tc, sc, stop_times_idx, rs_idx);

    for (auto const& trip : t_route.trips_) {
      utl::concat(tt->stop_times_, trip.stop_times_);
      utl::concat(tt->stop_attr_, trip.stop_attr_);
    }
    utl::concat(tt->route_stops_, t_route.route_stops_);
  }

  auto stop_times_idx = static_cast<stop_times_index>(tt->stop_times_.size());
  auto rs_idx = static_cast<route_stops_index>(tt->route_stops_.size());

  tt->routes_.emplace_back(0, 0, stop_times_idx, rs_idx);

  return tt;
}

auto create_route_map(transformable_timetable const& ttt) {
  route_mapping route_map{};

  for (route_id r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    auto& route = ttt.routes_[r_id];

    for (trip_id t_id = 0; t_id < route.trips_.size(); ++t_id) {
      auto const& el = route.trips_[t_id];
      route_map.insert_dbg(el.dbg_, r_id, t_id);
    }
  }

  return route_map;
}

auto get_station_departure_events(transformable_timetable const& ttt,
                                  stop_id const s_id) {
  std::vector<time> dep_events;

  auto const& station = ttt.stations_[s_id];
  for (auto const r_id : station.stop_routes_) {
    auto const& route = ttt.routes_[r_id];

    // - 1 since you cannot enter a route at the last stop
    for (auto offset = 0; offset < route.route_stops_.size() - 1; ++offset) {
      if (route.route_stops_[offset] != s_id) {
        continue;
      }

      for (auto const& trip : route.trips_) {
        if (!trip.lcons_[offset].in_allowed_) {
          continue;
        }
        dep_events.push_back(trip.lcons_[offset].departure_);
      }
    }
  }

  return dep_events;
}

auto get_initialization_footpaths(transformable_timetable const& ttt) {
  std::vector<std::vector<raptor_footpath>> init_footpaths(
      ttt.stations_.size());

  for (auto const& s : ttt.stations_) {
    for (auto const& f : s.footpaths_) {
      init_footpaths[f.from_].emplace_back(f.to_, f.duration_);
    }
  }

  return init_footpaths;
}

std::unique_ptr<raptor_meta_info> transformable_to_meta_info(
    transformable_timetable const& ttt) {
  auto meta_info = std::make_unique<raptor_meta_info>();

  // generate initialization footpaths BEFORE removing empty stations
  meta_info->initialization_footpaths_ = get_initialization_footpaths(ttt);

  meta_info->raptor_id_to_eva_.reserve(ttt.stations_.size());
  meta_info->station_id_to_index_.reserve(ttt.stations_.size());

  meta_info->departure_events_.resize(ttt.stations_.size());
  meta_info->equivalent_stations_.resize(ttt.stations_.size());

  // Loop over the stations
  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    auto const& s = ttt.stations_[s_id];

    meta_info->station_id_to_index_.push_back(s.motis_station_index_);
    meta_info->raptor_id_to_eva_.push_back(s.eva_);
    meta_info->eva_to_raptor_id_.emplace(
        s.eva_, static_cast<stop_id>(meta_info->eva_to_raptor_id_.size()));

    // set equivalent meta stations
    for (auto const equi_s_id : s.equivalent_) {
      meta_info->equivalent_stations_[s_id].push_back(equi_s_id);
    }

    // set departure events
    meta_info->departure_events_[s_id] =
        get_station_departure_events(ttt, s_id);

    // gather all departure events from stations reachable by foot
    for (auto const& f : ttt.stations_[s_id].footpaths_) {
      for (auto const& dep_event : get_station_departure_events(ttt, f.to_)) {
        meta_info->departure_events_[s_id].emplace_back(dep_event -
                                                        f.duration_);
      }
    }

    utl::erase_duplicates(meta_info->departure_events_[s_id]);
  }

  // create departure events with meta stations included
  meta_info->departure_events_with_metas_ = meta_info->departure_events_;

  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    auto& meta_departures = meta_info->departure_events_with_metas_[s_id];
    if (!meta_departures.empty()) {
      continue;
    }
    auto const s = ttt.stations_[s_id];
    for (auto const equi_s_id : s.equivalent_) {
      utl::concat(meta_departures, meta_info->departure_events_[equi_s_id]);
    }
    utl::erase_duplicates(meta_departures);
    for (auto const equi_s_id : s.equivalent_) {
      meta_info->departure_events_with_metas_[equi_s_id] = meta_departures;
    }
  }

  // Loop over the routes
  for (auto const& r : ttt.routes_) {
    for (auto const& t : r.trips_) {
      meta_info->lcon_ptr_.push_back(nullptr);
      for (auto const& rlc : t.lcons_) {
        meta_info->lcon_ptr_.push_back(rlc.lcon_);
      }
    }
  }

  meta_info->route_mapping_ = std::move(create_route_map(ttt));

  return meta_info;
}

std::pair<std::unique_ptr<raptor_meta_info>, std::unique_ptr<raptor_timetable>>
get_raptor_timetable(schedule const& sched) {
  log::scoped_timer timer("building RAPTOR timetable");

  transformable_timetable ttt;

  ttt.stations_.resize(sched.stations_.size());

  std::vector<std::thread> threads;
  threads.emplace_back(init_stops, std::cref(sched), std::ref(ttt.stations_));
  threads.emplace_back(init_routes, std::cref(sched), std::ref(ttt.routes_));
  threads.emplace_back(add_footpaths, std::cref(sched),
                       std::ref(ttt.stations_));

  std::for_each(begin(threads), end(threads), [](auto& t) { t.join(); });

  // after stops and routes are initialized
  init_stop_routes(ttt);

  LOG(log::info) << "RAPTOR Stations: " << ttt.stations_.size();
  LOG(log::info) << "RAPTOR Routes: " << ttt.routes_.size();

  auto footpaths = 0UL;
  for(auto& s : ttt.stations_)
    footpaths += s.incoming_footpaths_.size();
  LOG(log::info) << "RAPTOR Footpaths: " << footpaths;

  auto trips = 0UL;
  for(auto& t : ttt.routes_)
    trips += t.trips_.size();
  LOG(log::info) << "RAPTOR Trips: " << trips;

  auto meta_info = transformable_to_meta_info(ttt);
  auto tt = create_raptor_timetable(ttt);

  auto dep_events = 0UL;
  for(auto& s : meta_info->departure_events_)
    dep_events += s.size();
  LOG(log::info) << "RAPTOR Departure Events: " << dep_events;

  return {std::move(meta_info), std::move(tt)};
}

}  // namespace motis::raptor