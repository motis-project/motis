#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"

namespace motis::raptor {

using namespace motis::routing::output;

struct intermediate_journey {
  intermediate_journey(transfers const trs, bool const forward)
      : transfers_(trs), forward_(forward) {}

  time get_departure() const {
    return forward_ ? stops_.back().d_time_ : stops_.front().d_time_;
  }

  time get_arrival() const {
    return forward_ ? stops_.front().a_time_ : stops_.back().a_time_;
  }

  time get_duration() const { return get_arrival() - get_departure(); }

  void add_footpath(station_id const to, time const a_time, time const d_time,
                    time const duration, raptor_schedule const& raptor_sched) {
    auto const motis_index = raptor_sched.station_id_to_index_[to];
    if (forward_) {
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, a_time, d_time,
                          a_time, d_time, timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);
    } else {
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, -d_time, -a_time,
                          -d_time, -a_time, timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);
    }

    transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, 0, 0,
                             0);
  }

  time add_route(station_id const from, route_id const r_id, trip_id const trip,
                 stop_offset const exit_offset,
                 raptor_schedule const& raptor_sched,
                 raptor_timetable const& timetable) {
    auto const& route = timetable.routes_[r_id];

    auto const sti_base =
        route.index_to_stop_times_ + (trip * route.stop_count_);

    // Add the stops in backwards fashion, reverse the stop vector at the end
    for (auto s_offset = static_cast<int16_t>(exit_offset); s_offset >= 0;
         --s_offset) {
      auto const rsi = route.index_to_route_stops_ + s_offset;
      auto const s_id = timetable.route_stops_[rsi];

      auto const sti = sti_base + s_offset;
      auto const stop_time = timetable.stop_times_[sti];

      auto const d_time = stop_time.departure_;
      auto const tt = raptor_sched.transfer_times_[s_id];
      auto const a_time = stop_time.arrival_ - tt;

      if (s_id == from && d_time != 0) {
        return d_time;
      }

      auto const motis_index = raptor_sched.station_id_to_index_[s_id];

      if (forward_) {
        stops_.emplace_back(stops_.size(), motis_index, 0, 0, a_time, d_time,
                            a_time, d_time, timestamp_reason::SCHEDULE,
                            timestamp_reason::SCHEDULE, false, false);
      } else {
        stops_.emplace_back(stops_.size(), motis_index, 0, 0, -d_time, -a_time,
                            -d_time, -a_time, timestamp_reason::SCHEDULE,
                            timestamp_reason::SCHEDULE, false, false);
      }

      // We only have a single lcon_ptr array for the forward search,
      // therefore we need to adjust the index
      auto const backward_sti =
          sti_base + route.stop_count_ - 1 - (exit_offset - s_offset);
      auto const lcon = raptor_sched.lcon_ptr_[forward_ ? sti : backward_sti];
      transports_.emplace_back(stops_.size() - 1, stops_.size(), lcon);
    }

    LOG(motis::logging::warn)
        << "Could not correctly reconstruct RAPTOR journey";
    return invalid<time>;
  }

  void add_start_station(station_id const start,
                         raptor_schedule const& raptor_sched,
                         time const d_time) {
    auto const motis_index = raptor_sched.station_id_to_index_[start];
    if (forward_) {
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, INVALID_TIME,
                          d_time, INVALID_TIME, d_time,
                          timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);
    } else {
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, -d_time,
                          INVALID_TIME, -d_time, INVALID_TIME,
                          timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);
    }
  }

  journey to_journey(schedule const& sched) {
    journey j;

    if (forward_) {
      std::reverse(std::begin(stops_), std::end(stops_));
      std::reverse(std::begin(transports_), std::end(transports_));
    }

    unsigned idx = 0;
    for (auto& t : transports_) {
      t.from_ = idx;
      t.to_ = ++idx;
    }

    j.transports_ = generate_journey_transports(transports_, sched);
    j.trips_ = generate_journey_trips(transports_, sched);
    j.attributes_ = generate_journey_attributes(transports_);

    // HACK enter and exit flags TODO(julian)
    for (auto ts = 0; ts < transfers_ + 1; ++ts) {
      stops_[ts].enter_ = true;
      stops_[ts].exit_ = true;
    }

    stops_.front().a_time_ = INVALID_TIME;
    stops_.back().d_time_ = INVALID_TIME;

    j.stops_ = generate_journey_stops(stops_, sched);
    j.duration_ = 0;
    j.transfers_ = transfers_;
    j.db_costs_ = 0;
    j.price_ = 0;
    j.night_penalty_ = 0;
    return j;
  }

  transfers transfers_;
  bool forward_;
  std::vector<intermediate::stop> stops_;
  std::vector<intermediate::transport> transports_;
};

struct reconstructor {

  struct candidate {
    candidate() = default;
    candidate(station_id const source, station_id const target, time const dep,
              time const arr, transfers const t, bool const ends_with_footpath)
        : source_(source),
          target_(target),
          departure_(dep),
          arrival_(arr),
          transfers_(t),
          ends_with_footpath_(ends_with_footpath) {}

    bool dominates(candidate const& other) const {
      return arrival_ <= other.arrival_ && transfers_ <= other.transfers_;
    }

    station_id source_ = invalid<station_id>;
    station_id target_ = invalid<station_id>;

    time departure_ = invalid<time>;
    time arrival_ = invalid<time>;

    transfers transfers_ = invalid<transfers>;

    bool ends_with_footpath_ = false;
  };

  static std::string to_string(candidate const& c) {
    return "Dep: " + std::to_string(c.departure_) +
           " Arr: " + std::to_string(c.arrival_) +
           " Transfers: " + std::to_string(c.transfers_);
  }

  reconstructor() = delete;
  reconstructor(schedule const& sched, raptor_schedule const& raptor_sched,
                raptor_timetable const& tt)
      : sched_(sched), raptor_sched_(raptor_sched), timetable_(tt) {}

  static bool dominates(intermediate_journey const& ij, candidate const& c) {
    return (ij.get_arrival() <= c.arrival_ && ij.transfers_ <= c.transfers_);
  }

  static bool dominates(candidate const& c, intermediate_journey const& ij) {
    return (c.arrival_ < ij.get_arrival() && c.transfers_ <= ij.transfers_);
  }

  std::vector<candidate> get_candidates(raptor_query const& q) {
    auto const& result = *q.result_;

    std::vector<candidate> candidates;

    auto add_candidates = [&](station_id const t) {
      auto const tt = raptor_sched_.transfer_times_[t];

      for (auto round_k = 1; round_k < max_raptor_round; ++round_k) {
        if (!valid(result[round_k][t])) {
          continue;
        }

        candidate c(q.source_, t, q.source_time_begin_, result[round_k][t],
                    round_k - 1, true);

        // Check if the journey ends with a footpath
        for (; c.arrival_ < result[round_k][t] + tt; c.arrival_++) {
          c.ends_with_footpath_ = journey_ends_with_footpath(c, result);
          if (!c.ends_with_footpath_) {
            break;
          }
        }

        c.arrival_ -= tt;

        auto dominated = std::any_of(
            std::begin(candidates), std::end(candidates),
            [&](auto const& other_c) { return other_c.dominates(c); });

        dominated |=
            std::any_of(std::begin(journeys_), std::end(journeys_),
                        [&](auto const& j) { return dominates(j, c); });

        if (!dominated) {
          // Remove earlier candidates which are dominated by the new candidate
          erase_if(candidates,
                   [&](auto const& other_c) { return c.dominates(other_c); });

          candidates.push_back(c);
        }
      }
    };

    add_candidates(q.target_);

    if (q.use_dest_metas_) {
      for (auto equivalent : raptor_sched_.equivalent_stations_[q.target_]) {
        add_candidates(equivalent);
      }
    }

    return candidates;
  }

  [[maybe_unused]] static void print_c(std::ostream& out, candidate const& c) {
    out << "Candidate: " << c.departure_ << " " << c.arrival_ << " : "
        << std::to_string(c.transfers_) << '\n';
  }

  void add(raptor_query const& q) {
    for (auto& c : get_candidates(q)) {

      // We need to add the transfer time to the arrival,
      // since all arrivals in the results are with pre-added transfer times.
      // But only if the journey does not end with a footpath,
      // since footpaths have no pre-added transfer times.
      if (!c.ends_with_footpath_) {
        c.arrival_ += raptor_sched_.transfer_times_[c.target_];
      }

      journeys_.push_back(reconstruct_journey(c, *q.result_, q.forward_));
    }
  }

  auto get_journeys() {
    erase_if(journeys_, [&](auto const& ij) -> bool {
      return ij.get_duration() > max_travel_duration;
    });

    return to_vec(journeys_, [&](auto& ij) { return ij.to_journey(sched_); });
  }

  auto get_journeys(time const end) {
    erase_if(journeys_,
             [&](auto const& ij) -> bool { return ij.get_departure() > end; });

    return get_journeys();
  }

  bool journey_ends_with_footpath(candidate const c,
                                  raptor_result const& result) {
    auto const tuple =
        get_previous_station(c.target_, c.arrival_, c.transfers_ + 1, result);
    return !valid(std::get<0>(tuple));
  }

  intermediate_journey reconstruct_journey(candidate const c,
                                           raptor_result const& result,
                                           bool const forward) {
    intermediate_journey ij(c.transfers_, forward);

    auto arrival_station = c.target_;
    auto last_departure = invalid<time>;
    auto station_arrival = c.arrival_;
    for (auto result_idx = c.transfers_ + 1; result_idx > 0; --result_idx) {

      // std::cout << "result idx: " << result_idx << '\n';
      // std::cout << "last_departure: " << last_departure << "\n";
      // std::cout << "arrival station: " << arrival_station << '\n';
      // TODO(julian) possible to skip this if station arrival,
      // at index larger than last departure minus transfertime
      // same with footpath reachable stations
      auto [previous_station, used_route, used_trip, stop_offset] =
          get_previous_station(arrival_station, station_arrival, result_idx,
                               result);

      if (valid(previous_station)) {
        last_departure = ij.add_route(previous_station, used_route, used_trip,
                                      stop_offset, raptor_sched_, timetable_);
      } else {
        for (auto const& inc_f :
             timetable_.incoming_footpaths_[arrival_station]) {
          auto const adjusted_arrival = station_arrival - inc_f.duration_;
          std::tie(previous_station, used_route, used_trip, stop_offset) =
              get_previous_station(inc_f.from_, adjusted_arrival, result_idx,
                                   result);

          if (valid(previous_station)) {
            ij.add_footpath(arrival_station, station_arrival, last_departure,
                            inc_f.duration_, raptor_sched_);
            last_departure =
                ij.add_route(previous_station, used_route, used_trip,
                             stop_offset, raptor_sched_, timetable_);
            break;
          }
        }
      }

      // if (result_idx == 1) { break; }

      arrival_station = previous_station;
      station_arrival = result[result_idx - 1][arrival_station];
    }

    if (arrival_station == c.source_) {
      ij.add_start_station(c.source_, raptor_sched_, last_departure);
      return ij;
    }

    // We need to look for the start station
    auto const try_as_start = [&](station_id const start_station,
                                  station_id const to_station,
                                  time const last_departure) -> bool {
      for (auto const& inc_f : timetable_.incoming_footpaths_[to_station]) {
        if (inc_f.from_ != start_station) {
          continue;
        }

        ij.add_footpath(to_station, last_departure, last_departure,
                        inc_f.duration_, raptor_sched_);

        time const first_footpath_duration =
            inc_f.duration_ + raptor_sched_.transfer_times_[start_station];
        ij.add_start_station(start_station, raptor_sched_,
                             last_departure - first_footpath_duration);
        return true;
      }

      return false;
    };

    if (try_as_start(c.source_, arrival_station, last_departure)) {
      return ij;
    }

    for (auto const equi_s : raptor_sched_.equivalent_stations_[c.source_]) {
      if (try_as_start(equi_s, arrival_station, last_departure)) {
        return ij;
      }
    }

    return ij;
  }

  std::tuple<station_id, route_id, trip_id, stop_offset> get_previous_station(
      station_id const arrival_station, time const stop_arrival,
      uint8_t const result_idx, raptor_result const& result) {
    //  std::cout << "looked for previous station from: " << arrival_station
    //            << " with arrival: " << stop_arrival
    //            << " and actual arrival: " <<
    //            result[result_idx][arrival_station]
    //            << '\n';
    auto const arrival_stop = timetable_.stops_[arrival_station];

    auto const route_count = arrival_stop.route_count_;
    for (auto sri = arrival_stop.index_to_stop_routes_;
         sri < arrival_stop.index_to_stop_routes_ + route_count; ++sri) {

      auto const r_id = timetable_.stop_routes_[sri];
      auto const& route = timetable_.routes_[r_id];

      for (stop_offset offset = 1; offset < route.stop_count_; ++offset) {
        auto const rsi = route.index_to_route_stops_ + offset;
        auto const s_id = timetable_.route_stops_[rsi];
        if (s_id != arrival_station) {
          continue;
        }

        auto const arrival_trip =
            get_arrival_trip_at_station(r_id, stop_arrival, offset);

        if (!valid(arrival_trip)) {
          continue;
        }

        auto const board_station = get_board_station_for_trip(
            r_id, arrival_trip, result, result_idx - 1, offset);

        if (valid(board_station)) {
          // std::cout << "found one\n";
          return {board_station, r_id, arrival_trip, offset};
        }
      }
    }

    // std::cout << "Found NONE\n";
    return {invalid<station_id>, invalid<route_id>, invalid<trip_id>,
            invalid<stop_offset>};
  }

  trip_id get_arrival_trip_at_station(route_id const r_id, time const arrival,
                                      stop_offset const offset) {
    auto const& route = timetable_.routes_[r_id];

    for (auto trip = 0; trip < route.trip_count_; ++trip) {
      auto const sti =
          route.index_to_stop_times_ + (trip * route.stop_count_) + offset;
      if (timetable_.stop_times_[sti].arrival_ == arrival) {
        return trip;
      }
    }

    return invalid<trip_id>;
  }

  station_id get_board_station_for_trip(route_id const r_id, trip_id const t_id,
                                        raptor_result const& result,
                                        raptor_round const result_idx,
                                        stop_offset const arrival_offset) {
    auto const& r = timetable_.routes_[r_id];

    auto const first_stop_times_index =
        r.index_to_stop_times_ + (t_id * r.stop_count_);

    // -1, since we cannot board a trip at the last station
    auto const max_offset =
        std::min(static_cast<stop_offset>(r.stop_count_ - 1), arrival_offset);
    for (auto stop_offset = 0; stop_offset < max_offset; ++stop_offset) {
      auto const rsi = r.index_to_route_stops_ + stop_offset;
      auto const station_id = timetable_.route_stops_[rsi];

      auto const sti = first_stop_times_index + stop_offset;
      auto const departure = timetable_.stop_times_[sti].departure_;

      // if (r_id == 80081) {
      //   std::cout << "stop offset: " << stop_offset << '\n';
      //   std::cout << "station id: " << station_id << '\n';
      //   std::cout << "departure: " << departure << '\n';
      //   std::cout << "result[result_idx][station_id]: " <<
      //   result[result_idx][station_id] << '\n';
      // }

      if (valid(departure) && result[result_idx][station_id] <= departure) {
        return station_id;
      }
    }

    return invalid<station_id>;
  }

  schedule const& sched_;
  raptor_schedule const& raptor_sched_;
  raptor_timetable const& timetable_;

private:
  std::vector<intermediate_journey> journeys_;
};

}  // namespace motis::raptor