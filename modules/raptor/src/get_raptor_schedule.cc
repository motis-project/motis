#include "motis/raptor/get_raptor_schedule.h"

#include <numeric>
#include <thread>
#include <tuple>

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/raptor/raptor_util.h"

#include "utl/parallel_for.h"

namespace motis::raptor {

using namespace motis::logging;

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

std::vector<station_id> get_route_stops_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  std::vector<station_id> route_stops;
  route_stops.reserve(lcons.size() + 1);
  for (auto const& lcon : lcons) {
    route_stops.push_back(lcon.from_);
  }
  route_stops.push_back(lcons.back().to_);
  return route_stops;
}

time get_stand_time(transformable_route const& route) {
  time route_stand_time = invalid<time>;
  for (auto const& trip : route.trips_) {
    for (auto const& st : trip.stop_times_) {
      auto const stand_time = st.departure_ - st.arrival_;
      if (!valid(route_stand_time)) {
        route_stand_time = stand_time;
        continue;
      }

      if (stand_time != route_stand_time) {
        return invalid<time>;
      }
    }
  }
  return invalid<time>;
}

void init_stops(schedule const& sched, std::vector<transformable_stop>& tss) {
  for (auto s_id = 0; s_id < sched.stations_.size(); ++s_id) {
    // auto& s = ttt.stations_[s_id];
    auto& s = tss[s_id];
    auto const& s_ptr = sched.stations_[s_id];

    s.motis_station_index_ = s_ptr->index_;
    s.transfer_time_ = s_ptr->transfer_time_;
    s.eva_ = s_ptr->eva_nr_;

    for (auto equi_ptr : s_ptr->equivalent_) {
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
    sort_and_unique(station.stop_routes_);
  }
}

void init_routes(schedule const& sched, std::vector<transformable_route>& rs) {
  using namespace motis::access;

  // ttt.routes_.resize(sched.expanded_trips_.index_size() - 1);
  rs.resize(sched.expanded_trips_.index_size() - 1);

  route_id r_id = 0;
  for (auto const route_trips : sched.expanded_trips_) {
    // auto& t_route = ttt.routes_[r_id];
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

      ++t_id;
    }

    t_route.route_stops_ =
        get_route_stops_from_lcons(t_route.trips_.front().lcons_);
    t_route.stand_time_ = get_stand_time(t_route);

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
                   std::begin(raptor_station.footpaths_),
                   motis_fp_to_transformable);

    std::transform(std::begin(motis_station->incoming_footpaths_),
                   std::end(motis_station->incoming_footpaths_),
                   std::begin(raptor_station.incoming_footpaths_),
                   motis_fp_to_transformable);
  }
}

std::unique_ptr<raptor_timetable> create_raptor_timetable(
    transformable_timetable const& ttt) {
  auto tt = std::make_unique<raptor_timetable>();

  tt->stops_.reserve(ttt.stations_.size() + 1);
  tt->incoming_footpaths_.resize(ttt.stations_.size());

  for (auto s_id = 0U; s_id < ttt.stations_.size(); ++s_id) {
    auto const& t_stop = ttt.stations_[s_id];

    auto footpaths_idx = static_cast<footpaths_index>(tt->footpaths_.size());
    auto sr_idx = static_cast<stop_routes_index>(tt->stop_routes_.size());
    auto fc = static_cast<footpath_count>(t_stop.footpaths_.size());
    auto rc = static_cast<route_count>(t_stop.stop_routes_.size());
    tt->stops_.emplace_back(fc, rc, footpaths_idx, sr_idx);

    append_vector(tt->stop_routes_, t_stop.stop_routes_);

    for_each(t_stop.footpaths_, [&](auto const& f) {
      auto const transfer_time = ttt.stations_[f.from_].transfer_time_;
      tt->footpaths_.emplace_back(f.to_, f.duration_ - transfer_time);
    });

    for_each(t_stop.incoming_footpaths_, [&](auto const& f) {
      auto const transfer_time = ttt.stations_[f.from_].transfer_time_;
      tt->incoming_footpaths_[s_id].emplace_back(f.from_,
                                                 f.duration_ - transfer_time);
    });
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

    tt->routes_.emplace_back(tc, sc, stop_times_idx, rs_idx,
                             t_route.stand_time_);

    for (auto const& trip : t_route.trips_) {
      append_vector(tt->stop_times_, trip.stop_times_);
    }
    append_vector(tt->route_stops_, t_route.route_stops_);
  }

  auto stop_times_idx = static_cast<stop_times_index>(tt->stop_times_.size());
  auto rs_idx = static_cast<route_stops_index>(tt->route_stops_.size());

  tt->routes_.emplace_back(0, 0, stop_times_idx, rs_idx, invalid<time>);

  // preadd the transfer times
  for (auto const& route : tt->routes_) {
    for (auto trip = 0; trip < route.trip_count_; ++trip) {
      for (auto offset = 0; offset < route.stop_count_; ++offset) {
        auto const rsi = route.index_to_route_stops_ + offset;
        auto const sti =
            route.index_to_stop_times_ + (trip * route.stop_count_) + offset;

        auto const s_id = tt->route_stops_[rsi];
        auto const transfer_time = ttt.stations_[s_id].transfer_time_;
        auto& arrival = tt->stop_times_[sti].arrival_;
        if (valid(arrival)) {
          arrival += transfer_time;
        }
      }
    }
  }

  return tt;
}

std::unique_ptr<raptor_timetable> create_backward_raptor_timetable(
    transformable_timetable const& ttt, raptor_timetable const& tt) {
  auto btt = std::make_unique<raptor_timetable>(tt);

  // Adjust the routes for the backward timetable
  for (auto const& route : btt->routes_) {

    // Remove the preadded transfer time from the arrival values
    // Subtract it from the departure values, for correctness
    // Additionally, negate the time values
    for (auto trip = 0; trip < route.trip_count_; ++trip) {
      for (auto offset = 0; offset < route.stop_count_; ++offset) {
        auto const rsi = route.index_to_route_stops_ + offset;
        auto const sti =
            route.index_to_stop_times_ + (trip * route.stop_count_) + offset;
        auto const s_id = btt->route_stops_[rsi];
        auto const transfer_time = ttt.stations_[s_id].transfer_time_;

        auto& arrival = btt->stop_times_[sti].arrival_;
        auto& departure = btt->stop_times_[sti].departure_;

        if (valid(arrival)) {
          arrival = -(arrival - transfer_time);
        }
        if (valid(departure)) {
          departure = -(departure - transfer_time);
        }
        std::swap(arrival, departure);
      }
    }

    // Reverse the routes w.r.t trips and route stops
    auto st_from = std::begin(btt->stop_times_) + route.index_to_stop_times_;
    auto st_to = st_from + (route.trip_count_ * route.stop_count_);
    std::reverse(st_from, st_to);

    auto rs_from = std::begin(btt->route_stops_) + route.index_to_route_stops_;
    auto rs_to = rs_from + route.stop_count_;
    std::reverse(rs_from, rs_to);
  }

  // Generate footpaths for backwards timetable
  // Reverse the incoming footpaths for every station
  btt->footpaths_.clear();
  for (auto s_id = 0U; s_id < ttt.stations_.size(); ++s_id) {
    auto const& t_stop = ttt.stations_[s_id];

    auto footpaths_idx = static_cast<footpaths_index>(btt->footpaths_.size());
    auto fc = static_cast<footpath_count>(t_stop.incoming_footpaths_.size());
    btt->stops_[s_id].footpath_count_ = fc;
    btt->stops_[s_id].index_to_transfers_ = footpaths_idx;

    for_each(t_stop.incoming_footpaths_, [&](auto const& f) {
      auto const transfer_time = ttt.stations_[f.to_].transfer_time_;
      btt->footpaths_.emplace_back(f.from_, f.duration_ - transfer_time);
    });

    btt->incoming_footpaths_[s_id].clear();
    for_each(t_stop.footpaths_, [&](auto const& f) {
      auto const transfer_time = ttt.stations_[f.to_].transfer_time_;
      btt->incoming_footpaths_[s_id].emplace_back(f.to_,
                                                  f.duration_ - transfer_time);
    });
  }

  return btt;
}

auto get_station_departure_events(transformable_timetable const& ttt,
                                  station_id const s_id) {
  std::vector<time> dep_events;

  auto const& station = ttt.stations_[s_id];
  for (auto const r_id : station.stop_routes_) {
    auto const& route = ttt.routes_[r_id];

    for (auto offset = 0; offset < route.route_stops_.size(); ++offset) {
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

std::unique_ptr<raptor_schedule> transformable_to_schedule(
    transformable_timetable& ttt) {
  auto raptor_sched = std::make_unique<raptor_schedule>();

  // generate initialization footpaths BEFORE removing empty stations
  raptor_sched->initialization_footpaths_ = get_initialization_footpaths(ttt);

  raptor_sched->transfer_times_.reserve(ttt.stations_.size());
  raptor_sched->raptor_id_to_eva_.reserve(ttt.stations_.size());
  raptor_sched->station_id_to_index_.reserve(ttt.stations_.size());

  // raptor_sched->incoming_footpaths_.resize(ttt.stations_.size());
  raptor_sched->departure_events_.resize(ttt.stations_.size());
  raptor_sched->equivalent_stations_.resize(ttt.stations_.size());

  // Loop over the stations
  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    auto const& s = ttt.stations_[s_id];

    raptor_sched->station_id_to_index_.push_back(s.motis_station_index_);
    raptor_sched->transfer_times_.push_back(s.transfer_time_);
    raptor_sched->raptor_id_to_eva_.push_back(s.eva_);
    raptor_sched->eva_to_raptor_id_.emplace(
        s.eva_, raptor_sched->eva_to_raptor_id_.size());

    // set equivalent meta stations
    for (auto const equi_s_id : s.equivalent_) {
      raptor_sched->equivalent_stations_[s_id].push_back(equi_s_id);
    }

    // set departure events
    raptor_sched->departure_events_[s_id] =
        get_station_departure_events(ttt, s_id);

    // gather all departure events from stations reachable by foot
    for (auto const& f : ttt.stations_[s_id].footpaths_) {
      for_each(
          get_station_departure_events(ttt, f.to_), [&](auto const& dep_event) {
            raptor_sched->departure_events_[s_id].emplace_back(dep_event -
                                                               f.duration_);
          });
    }

    sort_and_unique(raptor_sched->departure_events_[s_id]);
  }

  // Loop over the routes
  for (auto const& r : ttt.routes_) {
    for (auto const& t : r.trips_) {
      raptor_sched->lcon_ptr_.push_back(nullptr);
      for (auto const& rlc : t.lcons_) {
        raptor_sched->lcon_ptr_.push_back(rlc.lcon_);
      }
    }
  }

  return raptor_sched;
}

// std::unique_ptr<raptor_schedule>
std::tuple<std::unique_ptr<raptor_schedule>, std::unique_ptr<raptor_timetable>,
           std::unique_ptr<raptor_timetable>>
get_raptor_schedule(schedule const& sched) {
  scoped_timer timer("building RAPTOR timetable");

  transformable_timetable ttt;

  ttt.stations_.resize(sched.stations_.size());

  std::vector<std::thread> threads;
  threads.emplace_back(init_stops, std::cref(sched), std::ref(ttt.stations_));
  threads.emplace_back(init_routes, std::cref(sched), std::ref(ttt.routes_));
  threads.emplace_back(add_footpaths, std::cref(sched),
                       std::ref(ttt.stations_));
  for (auto& t : threads) {
    t.join();
  }

  // after stops and routes are initialized
  init_stop_routes(ttt);

  LOG(info) << "RAPTOR Stations: " << ttt.stations_.size();
  LOG(info) << "RAPTOR Routes: " << ttt.routes_.size();

  auto raptor_sched = transformable_to_schedule(ttt);
  auto tt = create_raptor_timetable(ttt);
  auto btt = create_backward_raptor_timetable(ttt, *tt);

  return std::make_tuple(std::move(raptor_sched), std::move(tt),
                         std::move(btt));
}

}  // namespace motis::raptor