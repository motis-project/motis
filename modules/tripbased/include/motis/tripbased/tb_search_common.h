#pragma once

#include <array>
#include <functional>
#include <map>

#include "motis/core/common/logging.h"

#include "motis/tripbased/data.h"
#include "motis/tripbased/limits.h"
#include "motis/tripbased/tb_journey.h"
#include "motis/tripbased/tb_statistics.h"

namespace motis::tripbased {

enum class destination_mode { ALL, ANY };

struct queue_entry {
  queue_entry() = default;
  queue_entry(trip_id trip, stop_idx_t from_stop_index,
              stop_idx_t to_stop_index, std::size_t previous_trip_segment)
      : trip_(trip),
        from_stop_index_(from_stop_index),
        to_stop_index_(to_stop_index),
        previous_trip_segment_(previous_trip_segment) {}

  trip_id trip_{};
  stop_idx_t from_stop_index_{};
  stop_idx_t to_stop_index_{};
  std::size_t previous_trip_segment_{};
};

inline tb_reverse_transfer find_reverse_transfer(queue_entry const& from_qe,
                                                 queue_entry const& to_qe,
                                                 tb_data const& data) {
  auto const from_trip = from_qe.trip_;
  auto const to_trip = to_qe.trip_;
  auto const from_line = data.trip_to_line_[from_trip];
  auto const stop_count =
      std::min(from_qe.to_stop_index_,
               static_cast<stop_idx_t>(data.line_stop_count_[from_line] - 1));

  for (auto from_stop_idx = from_qe.from_stop_index_;
       from_stop_idx <= stop_count; ++from_stop_idx) {
    for (auto const& transfer : data.transfers_.at(from_trip, from_stop_idx)) {
      if (transfer.to_trip_ == to_trip) {
        if (transfer.to_stop_idx_ != to_qe.from_stop_index_) {
          continue;
        }
        return {from_trip, from_stop_idx,
                std::numeric_limits<stop_idx_t>::max()};
      }
    }
  }
  LOG(logging::error) << "trip-based journey reconstruction: find reverse "
                         "transfer (FWD) from trip "
                      << from_qe.trip_ << "."
                      << static_cast<int>(from_qe.to_stop_index_) << " to "
                      << to_qe.trip_ << "."
                      << static_cast<int>(to_qe.from_stop_index_)
                      << " not found";
  throw std::runtime_error{
      "trip-based journey reconstruction: transfer not found (fwd)"};
}

inline tb_transfer find_transfer(queue_entry const& from_qe,
                                 queue_entry const& to_qe,
                                 tb_data const& data) {
  auto const from_trip = from_qe.trip_;
  auto const to_trip = to_qe.trip_;
  auto const to_line = data.trip_to_line_[to_trip];
  for (auto to_stop_index = static_cast<int>(
           std::min(to_qe.to_stop_index_, data.line_stop_count_[to_line]));
       to_stop_index >= 0 /*to_qe.from_stop_index_*/; --to_stop_index) {
    for (auto const& transfer :
         data.reverse_transfers_.at(to_trip, to_stop_index)) {
      if (transfer.from_trip_ == from_trip) {
        if (transfer.from_stop_idx_ != from_qe.to_stop_index_) {
          continue;
        }
        return {to_trip, static_cast<stop_idx_t>(to_stop_index)};
      }
    }
  }
  LOG(logging::error)
      << "trip-based journey reconstruction: find transfer (BWD) from trip "
      << from_qe.trip_ << "." << static_cast<int>(from_qe.to_stop_index_)
      << " to " << to_qe.trip_ << ".["
      << static_cast<int>(to_qe.from_stop_index_) << ","
      << static_cast<int>(to_qe.to_stop_index_) << "] not found";

  throw std::runtime_error{
      "trip-based journey reconstruction: transfer not found (bwd)"};
}

template <search_dir Dir>
tb_footpath const& find_footpath(station_id from_station, station_id to_station,
                                 tb_data const& data, schedule const& sched) {
  for (auto const& fp : data.footpaths_[from_station]) {
    if (fp.to_stop_ == to_station) {
      return fp;
    }
  }
  LOG(logging::error) << "trip-based journey reconstruction: find footpath ("
                      << (Dir == search_dir::FWD ? "FWD" : "BWD")
                      << "): footpath from station " << from_station << " ("
                      << sched.stations_[from_station]->eva_nr_ << ": "
                      << sched.stations_[from_station]->name_ << " - "
                      << data.footpaths_[from_station].size() << "/"
                      << data.reverse_footpaths_[from_station].size()
                      << " footpaths) to " << to_station << " ("
                      << sched.stations_[to_station]->eva_nr_ << ": "
                      << sched.stations_[to_station]->name_ << " - "
                      << data.footpaths_[to_station].size() << "/"
                      << data.reverse_footpaths_[to_station].size()
                      << " footpaths) not found";
  throw std::runtime_error{
      "trip-based journey reconstruction: footpath not found"};
}

template <search_dir Dir>
void add_final_footpath(
    tb_journey& j, tb_data const& data, schedule const& sched,
    std::map<station_id, time> const& start_times,
    bool count_initial_transfer_time,
    std::function<bool(station_id)> const& is_start,
    std::function<duration_t(station_id)> const& get_initial_duration) {
  std::optional<tb_journey::tb_edge> best_fp;
  if (Dir == search_dir::FWD) {
    // edges:
    // ...
    // x -> y
    // start_station -> x
    // fp.from -> start_station
    auto const fp_arrival = j.edges_.back().departure_time_;
    time best_departure = std::numeric_limits<time>::min();
    if (is_start(j.start_station_)) {
      auto const fp_departure = static_cast<time>(
          fp_arrival - (count_initial_transfer_time
                            ? sched.stations_[j.start_station_]->transfer_time_
                            : 0));
      if (fp_departure >= start_times.at(j.start_station_)) {
        best_departure = fp_departure - get_initial_duration(j.start_station_);
      }
    }
    for (auto const& fp : data.reverse_footpaths_[j.start_station_]) {
      if (fp.is_interstation_walk() && is_start(fp.from_stop_)) {
        auto const fp_departure = static_cast<time>(fp_arrival - fp.duration_);
        auto const total_departure = static_cast<time>(
            fp_departure - get_initial_duration(fp.from_stop_));
        if (total_departure > best_departure &&
            fp_departure >= start_times.at(fp.from_stop_)) {
          best_departure = total_departure;
          best_fp = tb_journey::tb_edge{fp, fp_departure, fp_arrival};
        }
      }
    }

  } else {
    // edges:
    // ...
    // y -> x
    // x -> start_station
    // start_station -> fp.to
    auto const fp_departure = j.edges_.back().arrival_time_;
    time best_arrival = std::numeric_limits<time>::max();

    if (is_start(j.start_station_)) {
      auto const fp_arrival = static_cast<time>(
          fp_departure +
          (count_initial_transfer_time
               ? sched.stations_[j.start_station_]->transfer_time_
               : 0));
      if (fp_arrival <= start_times.at(j.start_station_)) {
        best_arrival = fp_arrival + get_initial_duration(j.start_station_);
      }
    }
    for (auto const& fp : data.footpaths_[j.start_station_]) {
      if (fp.is_interstation_walk() && is_start(fp.to_stop_)) {
        auto const fp_arrival = static_cast<time>(fp_departure + fp.duration_);
        auto const total_arrival =
            static_cast<time>(fp_arrival + get_initial_duration(fp.to_stop_));
        if (total_arrival < best_arrival &&
            fp_arrival <= start_times.at(fp.to_stop_)) {
          best_arrival = total_arrival;
          best_fp = tb_journey::tb_edge{fp, fp_departure, fp_arrival};
        }
      }
    }
  }
  if (best_fp.has_value()) {
    auto const& e = j.edges_.emplace_back(best_fp.value());
    j.start_station_ =
        Dir == search_dir::FWD ? e.footpath_.from_stop_ : e.footpath_.to_stop_;
  }
}

template <search_dir Dir>
void reconstruct_tb_journey(
    tb_journey& j, tb_data const& data, schedule const& sched,
    std::array<std::vector<queue_entry>, MAX_TRANSFERS + 1> const& queues,
    std::map<station_id, time> const& start_times,
    bool count_initial_transfer_time,
    std::function<bool(station_id)> const& is_start,
    std::function<duration_t(station_id)> const& get_initial_duration,
    tb_statistics& stats) {
  if (j.is_reconstructed()) {
    return;
  }
  ++stats.reconstruction_count_;
  auto queue_idx = j.final_queue_entry_;
  auto exit_stop_idx = j.destination_arrival_->stop_index_;
  auto const& dest_footpath = j.destination_arrival_->footpath_;
  if (dest_footpath.is_interstation_walk()) {
    if (Dir == search_dir::FWD) {
      auto const fp_arrival = j.arrival_time_;
      auto const fp_departure = fp_arrival - dest_footpath.duration_;
      j.edges_.emplace_back(dest_footpath, fp_departure, fp_arrival);
    } else {
      auto const fp_departure = j.arrival_time_;
      auto const fp_arrival = fp_departure + dest_footpath.duration_;
      j.edges_.emplace_back(dest_footpath, fp_departure, fp_arrival);
    }
  }
  for (auto t = static_cast<int>(j.transfers_); t >= 0; --t) {
    auto const& qe = queues[t][queue_idx];  // NOLINT
    auto const line = data.trip_to_line_[qe.trip_];

    if (Dir == search_dir::FWD) {
      j.edges_.emplace_back(
          qe.trip_, qe.from_stop_index_, exit_stop_idx,
          data.departure_times_[qe.trip_][qe.from_stop_index_],
          data.arrival_times_[qe.trip_][exit_stop_idx]);

      queue_idx = qe.previous_trip_segment_;
      if (t > 0) {
        auto transfer = find_reverse_transfer(
            queues[t - 1][qe.previous_trip_segment_], qe, data);  // NOLINT
        exit_stop_idx = transfer.from_stop_idx_;
        auto const cur_enter_station =
            data.stops_on_line_[line][qe.from_stop_index_];
        auto const prev_exit_station =
            data.stops_on_line_[data.trip_to_line_[transfer.from_trip_]]
                               [transfer.from_stop_idx_];
        if (prev_exit_station != cur_enter_station) {
          auto const& fp = find_footpath<Dir>(prev_exit_station,
                                              cur_enter_station, data, sched);
          auto const fp_departure =
              data.arrival_times_[transfer.from_trip_][transfer.from_stop_idx_];
          auto const fp_arrival =
              static_cast<time>(fp_departure + fp.duration_);
          j.edges_.emplace_back(fp, fp_departure, fp_arrival);
        }
      } else {
        j.start_station_ = data.stops_on_line_[data.trip_to_line_[qe.trip_]]
                                              [qe.from_stop_index_];
      }
    } else {
      j.edges_.emplace_back(qe.trip_, exit_stop_idx, qe.to_stop_index_,
                            data.departure_times_[qe.trip_][exit_stop_idx],
                            data.arrival_times_[qe.trip_][qe.to_stop_index_]);

      queue_idx = qe.previous_trip_segment_;
      if (t > 0) {
        auto transfer = find_transfer(
            qe, queues[t - 1][qe.previous_trip_segment_], data);  // NOLINT
        exit_stop_idx = transfer.to_stop_idx_;  // enter of next trip
        auto const cur_exit_station =
            data.stops_on_line_[line][qe.to_stop_index_];
        auto const next_enter_station =
            data.stops_on_line_[data.trip_to_line_[transfer.to_trip_]]
                               [transfer.to_stop_idx_];

        if (cur_exit_station != next_enter_station) {
          auto const& fp = find_footpath<Dir>(cur_exit_station,
                                              next_enter_station, data, sched);
          auto const fp_departure =
              data.arrival_times_[qe.trip_][qe.to_stop_index_];
          auto const fp_arrival =
              static_cast<time>(fp_departure + fp.duration_);
          j.edges_.emplace_back(fp, fp_departure, fp_arrival);
        }
      } else {
        j.start_station_ = data.stops_on_line_[data.trip_to_line_[qe.trip_]]
                                              [qe.to_stop_index_];
      }
    }
  }
  add_final_footpath<Dir>(j, data, sched, start_times,
                          count_initial_transfer_time, is_start,
                          get_initial_duration);
  if (Dir == search_dir::FWD) {
    std::reverse(begin(j.edges_), end(j.edges_));
  }
}

}  // namespace motis::tripbased