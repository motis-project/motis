#pragma once

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"

namespace motis::raptor {

using namespace motis::routing::output;

template <typename CriteriaConfig>
struct intermediate_journey {
  intermediate_journey(transfers const trs, bool const ontrip,
                       time const ontrip_start, trait_id const trait_offset)
      : transfers_{trs},
        ontrip_{ontrip},
        ontrip_start_{ontrip_start},
        trait_offset_{trait_offset} {}

  time get_departure() const { return stops_.back().d_time_; }
  time get_arrival() const { return stops_.front().a_time_; }

  time get_duration() const {
    return get_arrival() - (ontrip_ ? ontrip_start_ : get_departure());
  }

  void add_footpath(stop_id const to, time const a_time, time const d_time,
                    time const duration, raptor_meta_info const& raptor_sched) {
    auto const motis_index = raptor_sched.station_id_to_index_[to];
    stops_.emplace_back(stops_.size(), motis_index, 0, 0, 0, 0, a_time, d_time,
                        a_time, d_time, timestamp_reason::SCHEDULE,
                        timestamp_reason::SCHEDULE, false, true);
    transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, 0, 0,
                             0);
  }

  time add_route(stop_id const from, route_id const r_id,
                 trip_id const trip_idx, stop_offset const exit_offset,
                 raptor_meta_info const& raptor_sched,
                 raptor_timetable const& timetable) {
    auto const& route = timetable.routes_[r_id];

    auto const stop_time_idx_base =
        route.index_to_stop_times_ + (trip_idx * route.stop_count_);

    // Add the stops in backwards fashion, reverse the stop vector at the end
    for (auto s_offset = static_cast<int16_t>(exit_offset); s_offset >= 0;
         --s_offset) {
      auto const route_stop_idx = route.index_to_route_stops_ + s_offset;
      auto const station_idx = timetable.route_stops_[route_stop_idx];
      auto const stop_time_idx = stop_time_idx_base + s_offset;
      auto const stop_time = timetable.stop_times_[stop_time_idx];

      auto d_time = stop_time.departure_;
      auto a_time = stop_time.arrival_;
      // if (valid(a_time)) {
      //   a_time -= raptor_sched.transfer_times_[station_idx];
      // }

      if (station_idx == from && valid(d_time)) {
        return d_time;
      }

      // we is_exit at the last station -> d_time is invalid
      if (!valid(d_time) || s_offset == exit_offset) {
        if (transports_.empty() || transports_.back().is_walk()) {
          d_time = a_time;
        } else {
          d_time = transports_.back().con_->d_time_;
        }
      }

      auto const& get_d_track = [&](auto&& lcon) {
        if (transports_.empty() || transports_.back().is_walk()) {
          return lcon->full_con_->d_track_;
        } else {
          return transports_.back().con_->full_con_->d_track_;
        }
      };

      auto const lcon = raptor_sched.lcon_ptr_[stop_time_idx];
      auto const a_track = lcon->full_con_->a_track_;
      auto const d_track = get_d_track(lcon);

      if (!valid(a_time)) {
        a_time = lcon->a_time_;
      }

      auto const motis_index = raptor_sched.station_id_to_index_[station_idx];

      auto const is_enter = s_offset == exit_offset && !transports_.empty() &&
                            !transports_.back().is_walk();
      auto const is_exit = s_offset == exit_offset;

      stops_.emplace_back(stops_.size(), motis_index, a_track, d_track, a_track,
                          d_track, a_time, d_time, a_time, d_time,
                          timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, is_exit, is_enter);

      transports_.emplace_back(stops_.size() - 1, stops_.size(), lcon);
    }

    LOG(motis::logging::warn)
        << "Could not correctly reconstruct RAPTOR journey";
    return invalid<time>;
  }

  void add_start_station(stop_id const start,
                         raptor_meta_info const& raptor_sched,
                         time const d_time) {
    auto const motis_index = raptor_sched.station_id_to_index_[start];

    auto const enter = !transports_.empty() && !transports_.back().is_walk();
    auto const d_track =
        enter ? transports_.back().con_->full_con_->d_track_ : 0;

    stops_.emplace_back(stops_.size(), motis_index, 0, d_track, 0, d_track,
                        INVALID_TIME, d_time, INVALID_TIME, d_time,
                        timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                        false, enter);
  }

  void finalize() {
    std::reverse(std::begin(stops_), std::end(stops_));
    std::reverse(std::begin(transports_), std::end(transports_));

    unsigned idx = 0;
    for (auto& t : transports_) {
      t.from_ = idx;
      t.to_ = ++idx;
    }

    stops_.front().a_time_ = INVALID_TIME;
    stops_.back().d_time_ = INVALID_TIME;
  }

  journey to_journey(schedule const& sched) const {
    journey j;
    j.transports_ = generate_journey_transports(transports_, sched);
    j.trips_ = generate_journey_trips(transports_, sched);
    j.attributes_ = generate_journey_attributes(transports_);
    j.stops_ = generate_journey_stops(stops_, sched);
    j.duration_ = get_duration();
    j.transfers_ = transfers_;
    j.db_costs_ = 0;
    j.price_ = 0;
    j.night_penalty_ = 0;
    CriteriaConfig::fill_journey(j, trait_offset_);
    return j;
  }

  transfers transfers_;
  bool ontrip_;
  time ontrip_start_;
  std::vector<intermediate::stop> stops_;
  std::vector<intermediate::transport> transports_;
  trait_id trait_offset_;
};

template <typename CriteriaConfig>
struct reconstructor {
  using CriteriaData = CriteriaConfig;

  struct candidate {

    candidate() = default;
    candidate(stop_id const source, stop_id const target, time const dep,
              time const arr, transfers const t, trait_id trait_offset,
              bool const ends_with_footpath)
        : source_{source},
          target_{target},
          departure_{dep},
          arrival_{arr},
          transfers_{t},
          ends_with_footpath_{ends_with_footpath},
          trait_offset_{trait_offset} {}

    friend std::ostream& operator<<(std::ostream& out, candidate const& c) {
      return out << "(departure=" << c.departure_ << ", arrival=" << c.arrival_
                 << ", transfers=" << c.transfers_ << ")";
    }

    bool dominates(candidate const& other) const {
      return arrival_ <= other.arrival_ && transfers_ <= other.transfers_ &&
             CriteriaConfig::dominates(other.trait_offset_, trait_offset_);
    }

    stop_id source_{invalid<stop_id>};
    stop_id target_{invalid<stop_id>};

    time departure_{invalid<time>};
    time arrival_{invalid<time>};

    transfers transfers_ = invalid<transfers>;

    bool ends_with_footpath_ = false;

    trait_id trait_offset_{0};
  };

  reconstructor() = delete;

  reconstructor(schedule const& sched, raptor_meta_info const& raptor_sched,
                raptor_timetable const& tt)
      : sched_(sched), raptor_sched_(raptor_sched), timetable_(tt) {}

  static bool dominates(intermediate_journey<CriteriaConfig> const& ij,
                        candidate const& c) {
    return (ij.get_arrival() <= c.arrival_ && ij.transfers_ <= c.transfers_ &&
            CriteriaConfig::dominates(c.trait_offset_, ij.trait_offset_));
  }

  template <typename Query>
  std::vector<candidate> get_candidates(Query const& q) {
    auto const& result = q.result();

    std::vector<candidate> candidates;

    auto add_candidates = [&](stop_id const target) {

      for (auto round_k = 1; round_k < max_raptor_round; ++round_k) {
        for (trait_id trait_offset = 0,
                      trait_size = CriteriaConfig::TRAITS_SIZE;
             trait_offset < trait_size; ++trait_offset) {
          auto const arrival_idx =
              CriteriaConfig::get_arrival_idx(target, trait_offset);

          if (!valid(result[round_k][arrival_idx])) {
            continue;
          }

          auto c = candidate{q.source_,
                             target,
                             q.source_time_begin_,
                             result[round_k][arrival_idx],
                             static_cast<transfers>(round_k - 1),
                             trait_offset,
                             true};

          // Check if the journey ends with a footpath
          auto const tt = CriteriaConfig::get_transfer_time(
              timetable_, trait_offset, target);

          for (; c.arrival_ < result[round_k][arrival_idx] + tt; c.arrival_++) {
            c.ends_with_footpath_ = journey_ends_with_footpath(c, result);
            if (!c.ends_with_footpath_) {
              break;
            }
          }

          auto dominated = std::any_of(
              std::begin(candidates), std::end(candidates),
              [&](auto const& other_c) { return other_c.dominates(c); });

          dominated |=
              std::any_of(std::begin(journeys_), std::end(journeys_),
                          [&](auto const& j) { return dominates(j, c); });

          if (!dominated) {
            // Remove earlier candidates which are dominated by the new
            // candidate.
            utl::erase_if(candidates, [&](auto const& other_c) {
              return c.dominates(other_c);
            });

            candidates.push_back(c);
          }
        }
      }
    };

    if (!q.use_dest_metas_) {
      add_candidates(q.target_);
    } else {
      for (auto const& c : raptor_sched_.equivalent_stations_[q.target_]) {
        add_candidates(c);
      }
    }

    return candidates;
  }

  template <typename Query>
  void add(Query const& q) {
    for (auto& c : get_candidates(q)) {
      if (c.ends_with_footpath_) {
        // We need to add the transfer time to the arrival,
        // since all arrivals in the results are with pre-added transfer times.
        // But only if the journey does not end with a footpath,
        // since footpaths have no pre-added transfer times.
        // c.arrival_ -= timetable_.transfer_times_[c.target_];
        c.arrival_ -= CriteriaConfig::get_transfer_time(
            timetable_, c.trait_offset_, c.target_);
      }

      journeys_.push_back(reconstruct_journey(c, q));
    }
  }

  std::vector<journey> get_journeys() {
    utl::erase_if(journeys_, [&](auto const& ij) {
      return ij.get_duration() > max_travel_duration;
    });
    for (auto& ij : journeys_) {
      ij.finalize();
    }
    return utl::to_vec(journeys_,
                       [&](auto& ij) { return ij.to_journey(sched_); });
  }

  std::vector<journey> get_journeys(time const end) {
    utl::erase_if(journeys_, [&](auto const& ij) {
      return ij.get_departure() > end ||
             ij.get_duration() > max_travel_duration;
    });
    for (auto& ij : journeys_) {
      ij.finalize();
    }
    return utl::to_vec(journeys_,
                       [&](auto& ij) { return ij.to_journey(sched_); });
  }

  bool journey_ends_with_footpath(candidate const c,
                                  raptor_result_base const& result) {
    return !valid(std::get<stop_id>(get_previous_station(
        c.target_, c.arrival_, c.transfers_ + 1, c.trait_offset_, result)));
  }

  template <typename Query>
  intermediate_journey<CriteriaConfig> reconstruct_journey(candidate const c,
                                                           Query const& q) {
    auto const& result = q.result();

    auto ij = intermediate_journey<CriteriaConfig>{
        c.transfers_, q.ontrip_, q.source_time_begin_, c.trait_offset_};

    auto arrival_station = c.target_;
    auto last_departure = invalid<time>;
    auto station_arrival = c.arrival_;
    auto trait_offset = c.trait_offset_;
    for (auto result_idx = c.transfers_ + 1; result_idx > 0; --result_idx) {
      // than last departure minus transfertime,
      // same with footpath reachable stations
      auto [previous_station, used_route, used_trip, stop_offset,
            new_trait_offset] =
          get_previous_station(arrival_station, station_arrival, result_idx,
                               trait_offset, result);

      if (valid(previous_station)) {
        last_departure = ij.add_route(previous_station, used_route, used_trip,
                                      stop_offset, raptor_sched_, timetable_);
      } else {
        for (auto const& inc_f :
             timetable_.incoming_footpaths_[arrival_station]) {
          auto const adjusted_arrival = station_arrival - inc_f.duration_;
          std::tie(previous_station, used_route, used_trip, stop_offset,
                   new_trait_offset) =
              get_previous_station(inc_f.from_, adjusted_arrival, result_idx,
                                   trait_offset, result);

          if (valid(previous_station)) {
            auto const fp_arrival =
                station_arrival +
                CriteriaConfig::get_transfer_time(timetable_, trait_offset,
                                                  previous_station);
            ij.add_footpath(arrival_station, fp_arrival, last_departure,
                            inc_f.duration_, raptor_sched_);
            last_departure =
                ij.add_route(previous_station, used_route, used_trip,
                             stop_offset, raptor_sched_, timetable_);
            break;
          }
        }
      }

      arrival_station = previous_station;
      trait_offset = new_trait_offset;
      auto const arrival_idx =
          CriteriaConfig::get_arrival_idx(arrival_station, trait_offset);
      station_arrival = result[result_idx - 1][arrival_idx];
    }

    bool can_be_start = false;
    if (!q.use_start_metas_) {
      can_be_start = arrival_station == c.source_;
    } else {
      can_be_start = contains(raptor_sched_.equivalent_stations_[c.source_],
                              arrival_station);
    }

    if (can_be_start) {
      ij.add_start_station(arrival_station, raptor_sched_, last_departure);
    } else {
      // If we use start meta stations we need to search for the best one.
      // We need to search for the start meta station with the best departure
      // time. The best departure time is the latest one.

      auto const get_start_departure = [&](stop_id const start_station) {
        for (auto const& f : timetable_.incoming_footpaths_[arrival_station]) {
          if (f.from_ != start_station) {
            continue;
          }

          time const first_footpath_duration = f.duration_;

          return static_cast<time>(last_departure - first_footpath_duration);
        }

        return invalid<time>;
      };

      auto const add_start_with_footpath = [&](stop_id const start_station) {
        for (auto const& f : timetable_.incoming_footpaths_[arrival_station]) {
          if (f.from_ != start_station) {
            continue;
          }

          ij.add_footpath(arrival_station, last_departure, last_departure,
                          f.duration_, raptor_sched_);

          auto const first_footpath_duration = f.duration_;
          ij.add_start_station(start_station, raptor_sched_,
                               last_departure - first_footpath_duration);
        }
      };

      if (!q.use_start_metas_) {
        add_start_with_footpath(c.source_);
      } else {
        auto const& equivalents = raptor_sched_.equivalent_stations_[c.source_];
        auto const best_station = *std::max_element(
            std::begin(equivalents), std::end(equivalents),
            [&](auto const& s1, auto const& s2) -> bool {
              return get_start_departure(s1) < get_start_departure(s2);
            });
        add_start_with_footpath(best_station);
      }
    }

    return ij;
  }

  std::tuple<stop_id, route_id, trip_id, stop_offset, trait_id>
  get_previous_station(stop_id const arrival_station, time const stop_arrival,
                       uint8_t const result_idx, trait_id trait_offset,
                       raptor_result_base const& result) {
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

        auto const [board_station, new_trait_offset] =
            get_board_station_for_trip(r_id, arrival_trip, result,
                                       result_idx - 1, trait_offset, offset);

        if (valid(board_station)) {
          return {board_station, r_id, arrival_trip, offset, new_trait_offset};
        }
      }
    }

    return {invalid<stop_id>, invalid<route_id>, invalid<trip_id>,
            invalid<stop_offset>, invalid<trait_id>};
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

  std::tuple<stop_id, trait_id> get_board_station_for_trip(
      route_id const r_id, trip_id const t_id, raptor_result_base const& result,
      raptor_round const result_idx, trait_id const trait_offset,
      stop_offset const arrival_offset) {
    auto const& r = timetable_.routes_[r_id];

    auto const first_stop_times_index =
        r.index_to_stop_times_ + (t_id * r.stop_count_);

    // -1, since we cannot board a trip at the last station
    auto const max_offset =
        std::min(static_cast<stop_offset>(r.stop_count_ - 1), arrival_offset);
    CriteriaConfig aggregate{&r, trait_offset};

    for (auto stop_offset = 0; stop_offset < max_offset; ++stop_offset) {
      aggregate.reset(t_id, trait_offset);

      // 1. build the aggregate for this trip from offset 0 -> arr_offset
      for (auto s_offset = stop_offset + 1; s_offset <= max_offset;
           ++s_offset) {
        auto const current_sti = first_stop_times_index + s_offset;
        aggregate.update_from_stop(timetable_, s_offset, current_sti);
      }

      auto const rsi = r.index_to_route_stops_ + stop_offset;
      auto const stop_id = timetable_.route_stops_[rsi];

      auto const sti = first_stop_times_index + stop_offset;
      auto const departure = timetable_.stop_times_[sti].departure_;

      // while iterating search the trait offset downwards for possible
      // arrival times at the departure station. Thereby only iterate over
      // feasible trait offsets
      auto const feasible_trait_ids =
          result_idx > 0
              ? aggregate.get_feasible_traits(trait_offset)
              // the start values are not replicated along all trait offsets
              //  fall back to t_offset = 0 to find the start timings
              : std::vector<trait_id>{0};

      //we need to determine the transfer time based on the trait offset
      // the arrival was written on
      auto const transfer_time = CriteriaConfig::get_transfer_time(
          timetable_, trait_offset, stop_id);

      auto it = feasible_trait_ids.cbegin();
      while (it != feasible_trait_ids.cend()) {
        auto const arr_idx = CriteriaConfig::get_arrival_idx(stop_id, *it);

        if (valid(departure) &&
            result[result_idx][arr_idx] + transfer_time <= departure) {
          return {stop_id, *it};
        }

        ++it;
      }
    }

    return {invalid<stop_id>, invalid<trait_id>};
  }

private:
  schedule const& sched_;
  raptor_meta_info const& raptor_sched_;
  raptor_timetable const& timetable_;
  std::vector<intermediate_journey<CriteriaConfig>> journeys_;
};

}  // namespace motis::raptor
