#pragma once

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "motis/mcraptor/raptor_query.h"
#include "motis/mcraptor/raptor_timetable.h"
#include "motis/mcraptor/raptor_util.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"
#include "label.h"

namespace motis::mcraptor {

using namespace motis::routing::output;

struct intermediate_journey {
  intermediate_journey(transfers const trs, bool const ontrip,
                       time const ontrip_start)
      : transfers_{trs}, ontrip_{ontrip}, ontrip_start_{ontrip_start} {}

  time get_departure() const { return stops_.back().d_time_; }
  time get_arrival() const { return stops_.front().a_time_; }

  time get_duration() const {
    return get_arrival() - (ontrip_ ? ontrip_start_ : get_departure());
  }

  void add_footpath(stop_id const to, time const a_time, time const d_time,
                    uint16_t const d_track, time const duration,
                    raptor_meta_info const& raptor_sched, bool exit = false,
                    bool enter = true) {
    auto const motis_index = raptor_sched.station_id_to_index_[to];
    stops_.emplace_back(stops_.size(), motis_index, 0, d_track, 0, d_track,
                        a_time + duration, d_time, a_time + duration, d_time,
                        timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                        exit, enter);
    transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, -1, 0,
                             0);
  }

  void add_start_footpath(stop_id to, time to_d_time, time const duration,
                          raptor_meta_info const& raptor_sched) {
    auto const motis_index_to = raptor_sched.station_id_to_index_[to];
    auto const motis_index_START = raptor_sched.station_id_to_index_[0];
    auto const enter = !transports_.empty() && !transports_.back().is_walk();
    auto const d_track =
        enter ? transports_.back().con_->full_con_->d_track_ : 0;
    stops_.emplace_back(stops_.size(), motis_index_to, 0, d_track, 0, d_track,
                        to_d_time - 5, to_d_time, to_d_time - 5, to_d_time,
                        timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                        false, enter);
    stops_.emplace_back(stops_.size(), motis_index_START, 0, 0, 0, 0,
                        INVALID_TIME, to_d_time - duration - 5, INVALID_TIME,
                        to_d_time - duration - 5, timestamp_reason::SCHEDULE,
                        timestamp_reason::SCHEDULE, false, false);
    transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, 0, 0,
                             0);
  }

  void add_end_footpath(time const a_time, time const duration,
                        raptor_meta_info const& raptor_sched) {
    auto const motis_index = raptor_sched.station_id_to_index_[1];
    stops_.emplace_back(stops_.size(), motis_index, 0, 0, 0, 0, a_time,
                        INVALID_TIME, a_time, INVALID_TIME,
                        timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                        false, false);
    transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, 0, 0,
                             0);
  }

  std::pair<time, uint16_t> add_route(stop_id const from, route_id const r_id,
                                      trip_id const trip_idx,
                                      stop_offset const exit_offset,
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
      if (valid(a_time)) {
        a_time -= raptor_sched.transfer_times_[station_idx];
      }

      auto const& get_d_track = [&](auto&& lcon) {
        if (transports_.empty() || transports_.back().is_walk() ||
            transports_.back().mumo_id_ == -1) {
          return lcon->full_con_->d_track_;
        } else {
          return transports_.back().con_->full_con_->d_track_;
        }
      };
      auto const lcon = raptor_sched.lcon_ptr_[stop_time_idx];
      auto const d_track = get_d_track(lcon);

      if (station_idx == from && valid(d_time)) {
        return std::pair<time, uint16_t>(d_time, d_track);
      }

      // we is_exit at the last station -> d_time is invalid
      if (!valid(d_time) || s_offset == exit_offset) {
        if (transports_.empty() || transports_.back().is_walk() ||
            transports_.back().mumo_id_ == -1) {
          d_time = a_time;
        } else {
          d_time = transports_.back().con_->d_time_;
        }
      }

      if(lcon == NULL) {
        return std::pair<time, uint16_t>(invalid<time>, invalid<uint16_t>);
      }
      auto const a_track = lcon->full_con_->a_track_;
      auto const motis_index = raptor_sched.station_id_to_index_[station_idx];

      if (!valid(a_time)) {
        a_time = lcon->a_time_;
      }

      auto const is_enter = s_offset == exit_offset && !transports_.empty() &&
                            !transports_.back().is_walk();
      auto const is_exit = s_offset == exit_offset;

      if (is_exit && !is_enter && !transports_.empty() &&
          transports_.back().is_walk() && transports_.back().mumo_id_ == -1 &&
          !stops_.empty() && stops_.back().enter_ &&
          stops_.back().station_id_ == motis_index) {
        auto& same_stop = stops_.back();
        same_stop.a_time_ = a_time;
        same_stop.a_sched_time_ = a_time;
        same_stop.a_track_ = a_track;
        same_stop.a_sched_track_ = a_track;
        same_stop.a_reason_ = timestamp_reason::SCHEDULE;
        same_stop.exit_ = true;
        transports_.pop_back();
      } else {
        stops_.emplace_back(stops_.size(), motis_index, a_track, d_track,
                            a_track, d_track, a_time, d_time, a_time, d_time,
                            timestamp_reason::SCHEDULE,
                            timestamp_reason::SCHEDULE, is_exit, is_enter);
      }
      transports_.emplace_back(stops_.size() - 1, stops_.size(), lcon);
    }

    LOG(motis::logging::warn)
        << "Could not correctly reconstruct RAPTOR journey";
    return std::pair<time, uint16_t>(invalid<time>, invalid<uint16_t>);
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
    return j;
  }

  transfers transfers_;
  bool ontrip_;
  time ontrip_start_;
  std::vector<intermediate::stop> stops_;
  std::vector<intermediate::transport> transports_;
};

struct reconstructor {
  struct candidate {
    candidate() = default;
    candidate(stop_id const source, stop_id const target, time const dep,
              time const arr, transfers const t, bool const ends_with_footpath)
        : source_{source},
          target_{target},
          departure_{dep},
          arrival_{arr},
          transfers_{t},
          ends_with_footpath_{ends_with_footpath} {}

    friend std::ostream& operator<<(std::ostream& out, candidate const& c) {
      return out << "(departure=" << c.departure_ << ", arrival=" << c.arrival_
                 << ", transfers=" << c.transfers_ << ")";
    }

    bool dominates(candidate const& other) const {
      return arrival_ <= other.arrival_ && transfers_ <= other.transfers_;
    }

    stop_id source_{invalid<stop_id>};
    stop_id target_{invalid<stop_id>};

    time departure_{invalid<time>};
    time arrival_{invalid<time>};

    transfers transfers_ = invalid<transfers>;

    bool ends_with_footpath_ = false;
  };

  reconstructor() = delete;

  reconstructor(schedule const& sched, raptor_meta_info const& raptor_sched,
                raptor_timetable const& tt)
      : sched_(sched), raptor_sched_(raptor_sched), timetable_(tt) {}

  static bool dominates(intermediate_journey const& ij, candidate const& c) {
    return (ij.get_arrival() <= c.arrival_ && ij.transfers_ <= c.transfers_);
  }

  template <class L>
  void add(raptor_query<L> const& q) {
    rounds<L>& result = q.result();
    L empty_label;
    bag<L> filter_bag;
    for (raptor_edge target_edge: q.raptor_edges_end_) {
      auto labels =
          result.getAllLabelsForStop(target_edge.from_, max_raptor_round * 2, false);
      for (L& label : labels) {
        if (label.is_in_range(q.source_time_begin_, q.source_time_end_)) {
          label.current_target_ = target_edge.from_;
          filter_bag.merge(label, true);
//          label.out();
//          filter_bag.labels_.push_back(label);
        }
      }
    }

    for (L& l: filter_bag.labels_) {
      intermediate_journey ij = intermediate_journey(
          l.changes_count_, q.ontrip_, q.source_time_begin_);

      if (q.target_ == 1) {
        for (raptor_edge edge : q.raptor_edges_end_) {
          if (edge.from_ == l.current_target_) {
            time8 min_footpath_duration = edge.duration_;
            auto index_into_transfers = q.tt_.stops_[edge.from_].index_to_transfers_;
            auto next_index_into_transfers = q.tt_.stops_[edge.from_ + 1].index_to_transfers_;
            for (raptor_edge target_edge_to: q.raptor_edges_end_) {
              for (auto current_index = index_into_transfers;
                   current_index < next_index_into_transfers; ++current_index) {
                auto const& to_stop = q.tt_.footpaths_[current_index].to_;
                auto const& duration = q.tt_.footpaths_[current_index].duration_;
                if (to_stop == target_edge_to.from_) {
                  if (duration + target_edge_to.duration_ < min_footpath_duration) {
                    min_footpath_duration = duration + target_edge_to.duration_;
                  }
                  break;
                }
              }
            }
            ij.add_end_footpath(l.arrival_time_ - edge.duration_ + min_footpath_duration,
                                min_footpath_duration, raptor_sched_);
            break;
          }
        }
      }

      L current_station_label = l;
      L target_station_label = l;
      raptor_round r_k = current_station_label.changes_count_;
      stop_id current_station = l.current_target_;
      stop_id parent_station = current_station_label.parent_station_;
      std::pair<time, uint16_t> last_departure_info =
          std::pair<time, uint16_t>(invalid<time>, invalid<uint16_t>);
      bool invalid_path = false;
      while (r_k > 0) {
        if (r_k == 1 && q.source_ != 0) {
          if (std::find(
                  q.meta_info_.equivalent_stations_.at(parent_station).begin(),
                  q.meta_info_.equivalent_stations_.at(parent_station).end(),
                  current_station) !=
              q.meta_info_.equivalent_stations_.at(parent_station).end()) {
            break;
          }
        }
        if (r_k % 2 == 0 && current_station_label.route_id_) {
          if (q.source_ != 0) {
            raptor_route route = q.tt_.routes_[current_station_label.route_id_];
            stop_id stop =
                q.tt_.route_stops_[route.index_to_route_stops_ +
                                   current_station_label.stop_offset_];
            stop_id mid = invalid<stop_id>;
            if (std::find(q.meta_info_.equivalent_stations_[q.target_].begin(),
                          q.meta_info_.equivalent_stations_[q.target_].end(),
                          stop) ==
                q.meta_info_.equivalent_stations_[q.target_].end()) {
              for (stop_id s : q.meta_info_.equivalent_stations_[q.target_]) {
                for (stop_id st : q.meta_info_.equivalent_stations_[stop]) {
                  if (st == s && s == q.target_) {
                    continue;
                  }
                  if (st == s) {
                    mid = s;
                    break;
                  }
                }
              }
              if (mid != invalid<stop_id>) {
                auto index_into_transfers =
                    q.tt_.stops_[stop].index_to_transfers_;
                auto next_index_into_transfers =
                    q.tt_.stops_[stop + 1].index_to_transfers_;
                for (auto current_index = index_into_transfers;
                     current_index < next_index_into_transfers;
                     ++current_index) {
                  auto const& to_stop = q.tt_.footpaths_[current_index].to_;
                  auto const& duration =
                      q.tt_.footpaths_[current_index].duration_;
                  if (to_stop == mid) {
                    ij.add_footpath(
                        to_stop, current_station_label.arrival_time_,
                        last_departure_info.first, last_departure_info.second,
                        duration, raptor_sched_, false, false);
                    last_departure_info = std::pair<time, uint16_t>(
                        last_departure_info.first -
                            current_station_label.footpath_duration_,
                        invalid<uint16_t>);
                    break;
                  }
                }
              }
            }
          }
          last_departure_info = ij.add_route(
              parent_station, current_station_label.route_id_,
              current_station_label.current_trip_id_,
              current_station_label.stop_offset_, raptor_sched_, timetable_);
          if (!valid(last_departure_info.first)) {
            invalid_path = true;
            //          std::cout << "Invalid path worked!" << std::endl;
            break;
          }
        } else if (r_k % 2 == 1 && r_k != target_station_label.changes_count_ &&
                   valid(current_station_label.footpath_duration_)) {
          ij.add_footpath(current_station, current_station_label.arrival_time_,
                          last_departure_info.first, last_departure_info.second,
                          current_station_label.footpath_duration_,
                          raptor_sched_);
          last_departure_info = std::pair<time, uint16_t>(
              last_departure_info.first -
                  current_station_label.footpath_duration_,
              invalid<uint16_t>);
        }

        r_k--;
        current_station_label = result[r_k][parent_station].get_fastest_label(
            last_departure_info.first, empty_label);
        if (!valid(current_station_label.journey_departure_time_)) {
          invalid_path = true;
//          std::cout << "Invalid path worked!" << std::endl;
          break;
        }
        current_station = parent_station;
        parent_station = current_station_label.parent_station_;
      }

      if (invalid_path) {
        continue;
      }

      if (q.source_ == 0) {
        for (raptor_edge edge : q.raptor_edges_start_) {
          if (edge.to_ == current_station) {
            ij.add_start_footpath(current_station, last_departure_info.first,
                                  edge.duration_, raptor_sched_);
            break;
          }
        }
      } else {
        ij.add_start_station(current_station, raptor_sched_,
                             last_departure_info.first);
      }
      journeys_.push_back(ij);
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

private:
  schedule const& sched_;
  raptor_meta_info const& raptor_sched_;
  raptor_timetable const& timetable_;
  std::vector<intermediate_journey> journeys_;
};

}  // namespace motis::mcraptor
