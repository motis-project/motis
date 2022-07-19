#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <queue>
#include <vector>

#include "utl/erase_if.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/edges.h"

#include "motis/tripbased/data.h"
#include "motis/tripbased/limits.h"
#include "motis/tripbased/tb_journey.h"
#include "motis/tripbased/tb_search_common.h"
#include "motis/tripbased/tb_statistics.h"

namespace motis::tripbased {

template <search_dir Dir = search_dir::FWD>
struct tb_ontrip_search {
  static constexpr time INVALID = Dir == search_dir::FWD
                                      ? std::numeric_limits<time>::max()
                                      : std::numeric_limits<time>::min();

  tb_ontrip_search(tb_data const& data, schedule const& sched, time start_time,
                   bool count_initial_transfer_time,
                   bool count_final_transfer_time, destination_mode dest_mode)
      : data_(data),
        sched_(sched),
        start_time(start_time),
        count_initial_transfer_time_(count_initial_transfer_time),
        count_final_transfer_time_(count_final_transfer_time),
        destination_mode_(dest_mode),
        destination_arrivals_(data.line_count_),
        first_reachable_stop_(data.trip_idx_end_,
                              Dir == search_dir::FWD
                                  ? std::numeric_limits<stop_idx_t>::max()
                                  : std::numeric_limits<stop_idx_t>::min()) {}

  void add_start(station_id stop_id, time initial_duration,
                 bool allow_footpaths = true) {
    assert(stop_id < sched_.stations_.size());
    auto const station_arrival = Dir == search_dir::FWD
                                     ? start_time + initial_duration
                                     : start_time - initial_duration;
    start_stations_.push_back(stop_id);
    start_times_[stop_id] = station_arrival;
    add_start(
        {stop_id, stop_id,
         count_initial_transfer_time_
             ? static_cast<unsigned>(sched_.stations_[stop_id]->transfer_time_)
             : 0},
        initial_duration);
    if (allow_footpaths) {
      auto const footpaths = Dir == search_dir::FWD
                                 ? data_.footpaths_[stop_id]
                                 : data_.reverse_footpaths_[stop_id];
      for (auto const& fp : footpaths) {
        add_start(fp, initial_duration);
      }
    }
    ++stats_.start_count_;
  }

  void add_destination(station_id stop_id, bool allow_footpaths = true) {
    assert(stop_id < sched_.stations_.size());
    destination_stations_.push_back(stop_id);
    add_destination(
        {stop_id, stop_id,
         count_final_transfer_time_
             ? static_cast<unsigned>(sched_.stations_[stop_id]->transfer_time_)
             : 0});
    if (allow_footpaths) {
      auto const footpaths = Dir == search_dir::FWD
                                 ? data_.reverse_footpaths_[stop_id]
                                 : data_.footpaths_[stop_id];
      for (auto const& fp : footpaths) {
        add_destination(fp);
      }
    }
    ++stats_.destination_count_;
  }

  void search() {
    journeys_.resize(sched_.stations_.size());
    earliest_arrival_.resize(sched_.stations_.size(), INVALID);

    add_direct_walks();

    for (auto transfers = 0U; transfers < MAX_TRANSFERS; ++transfers) {
      auto& queue = queues_[transfers];  // NOLINT
      if (!queue.empty()) {
        ++stats_.queue_count_;
      }
      if (Dir == search_dir::FWD) {
        search_fwd(transfers, queue);
      } else {
        search_bwd(transfers, queue);
      }
    }

    assert(queues_.size() == stats_.queue_size_.size());
    for (auto i = 0UL; i < queues_.size(); ++i) {
      auto const size = queues_[i].size();  // NOLINT
      stats_.queue_size_[i] = size;  // NOLINT
      stats_.max_queue_size_ =
          std::max(stats_.max_queue_size_, static_cast<uint64_t>(size));
    }
  }

  std::vector<tb_journey>& get_results(station_id destination,
                                       bool reconstruct = true) {
    auto& journeys = journeys_[destination];
    if (reconstruct) {
      for (auto& j : journeys) {
        reconstruct_journey(j);
      }
    }
    return journeys;
  }

  void reconstruct_journey(tb_journey& j) {
    reconstruct_tb_journey<Dir>(
        j, data_, sched_, queues_, start_times_, count_initial_transfer_time_,
        [this](station_id station) { return is_start(station); },
        [this](station_id station) { return get_initial_duration(station); },
        stats_);
  }

  tb_statistics& get_statistics() { return stats_; }
  tb_statistics const& get_statistics() const { return stats_; }

private:
  void add_start(tb_footpath const& fp, time initial_duration) {
    auto const offset = initial_duration + fp.duration_;
    auto const arrival_time = static_cast<time>(
        Dir == search_dir::FWD ? start_time + offset : start_time - offset);
    auto const start_stop =
        Dir == search_dir::FWD ? fp.to_stop_ : fp.from_stop_;
    for (auto const& [line, stop_idx, _] : data_.lines_at_stop_[start_stop]) {
      (void)_;
      auto const allowed = Dir == search_dir::FWD
                               ? data_.in_allowed_[line][stop_idx]
                               : data_.out_allowed_[line][stop_idx];
      if (allowed == 0U) {
        continue;
      }
      if ((Dir == search_dir::FWD &&
           stop_idx == data_.line_stop_count_[line] - 1) ||
          (Dir == search_dir::BWD && stop_idx == 0)) {
        continue;
      }
      auto const trip =
          Dir == search_dir::FWD
              ? data_.first_reachable_trip(line, stop_idx, arrival_time)
              : data_.last_reachable_trip(line, stop_idx, arrival_time);
      if (trip) {
        enqueue(trip->first, stop_idx, 0, 0);
      }
    }
  }

  void add_destination(tb_footpath const& fp) {
    auto const dest_stop = Dir == search_dir::FWD ? fp.from_stop_ : fp.to_stop_;
    for (auto const& [line, stop_idx, _] : data_.lines_at_stop_[dest_stop]) {
      (void)_;
      auto const allowed = Dir == search_dir::FWD
                               ? data_.out_allowed_[line][stop_idx]
                               : data_.in_allowed_[line][stop_idx];
      if (allowed == 0U) {
        continue;
      }
      destination_arrivals_[line].emplace_back(line, stop_idx, fp);
    }
  }

  void add_direct_walks() {
    for (auto const& start : start_stations_) {
      for (auto const& destination : destination_stations_) {
        add_direct_footpath(start, destination);
      }
    }
  }

  void add_direct_footpath(station_id start, station_id destination) {
    if (Dir == search_dir::FWD) {
      for (auto const& fp : data_.footpaths_[start]) {
        if (fp.to_stop_ == destination) {
          auto const departure_time = start_times_[start];
          auto const arrival_time =
              static_cast<time>(departure_time + fp.duration_);
          tb_journey j{Dir, start_time,  arrival_time, 0,
                       0,   destination, nullptr,      0};
          j.start_station_ = start;
          j.edges_.emplace_back(fp, departure_time, arrival_time);
          add_result(journeys_[destination], j);
          break;
        }
      }
    } else {
      for (auto const& fp : data_.reverse_footpaths_[start]) {
        if (fp.from_stop_ == destination) {
          auto const arrival_time = start_times_[start];
          auto const departure_time =
              static_cast<time>(arrival_time - fp.duration_);
          tb_journey j{Dir, start_time,  arrival_time, 0,
                       0,   destination, nullptr,      0};
          j.start_station_ = start;
          j.edges_.emplace_back(fp, departure_time, arrival_time);
          add_result(journeys_[destination], j);
          break;
        }
      }
    }
  }

  inline void search_fwd(unsigned const transfers,
                         std::vector<queue_entry>& queue) {
    for (auto current_trip_segment = 0UL; current_trip_segment < queue.size();
         ++current_trip_segment) {
      ++stats_.trip_segments_scanned_;
      auto& entry = queue[current_trip_segment];
      auto const line = data_.trip_to_line_[entry.trip_];
      auto const& destination_arrivals = destination_arrivals_[line];
      if (!destination_arrivals.empty()) {
        ++stats_.lines_reaching_destination_;
        stats_.destination_arrivals_scanned_ += destination_arrivals.size();
        for (auto const& dest_arrival : destination_arrivals) {
          if (entry.from_stop_index_ >= dest_arrival.stop_index_) {
            continue;
          }
          auto const arrival_time = static_cast<time>(
              data_.arrival_times_[entry.trip_][dest_arrival.stop_index_] +
              dest_arrival.footpath_.duration_);
          destination_reached(arrival_time, transfers, dest_arrival,
                              current_trip_segment);
        }
      }

      auto const next_stop_arrival =
          data_.arrival_times_[entry.trip_][entry.from_stop_index_ + 1];
      if (next_stop_arrival >= total_earliest_arrival_) {
        ++stats_.pruned_by_earliest_arrival_;
        continue;
      }
      auto const stop_count =
          std::min(entry.to_stop_index_,
                   static_cast<stop_idx_t>(data_.line_stop_count_[line] - 1));
      for (auto stop_idx = entry.from_stop_index_ + 1; stop_idx <= stop_count;
           ++stop_idx) {
        for (auto const& transfer :
             data_.transfers_.at(entry.trip_, stop_idx)) {
          ++stats_.transfers_scanned_;
          enqueue(transfer.to_trip_, transfer.to_stop_idx_, transfers + 1,
                  current_trip_segment);
        }
      }
    }
  }

  inline void search_bwd(unsigned const transfers,
                         std::vector<queue_entry>& queue) {
    for (auto current_trip_segment = 0UL; current_trip_segment < queue.size();
         ++current_trip_segment) {
      ++stats_.trip_segments_scanned_;
      auto& entry = queue[current_trip_segment];
      auto const line = data_.trip_to_line_[entry.trip_];
      auto const& destination_arrivals = destination_arrivals_[line];
      if (!destination_arrivals.empty()) {
        ++stats_.lines_reaching_destination_;
        stats_.destination_arrivals_scanned_ += destination_arrivals.size();
        for (auto const& dest_arrival : destination_arrivals) {
          if (entry.to_stop_index_ <= dest_arrival.stop_index_) {
            continue;
          }
          auto const arrival_time = static_cast<time>(
              data_.departure_times_[entry.trip_][dest_arrival.stop_index_] -
              dest_arrival.footpath_.duration_);
          destination_reached(arrival_time, transfers, dest_arrival,
                              current_trip_segment);
        }
      }

      assert(entry.to_stop_index_ > 0);
      auto const prev_stop_departure =
          data_.departure_times_[entry.trip_][entry.to_stop_index_ - 1];
      if (prev_stop_departure <= total_earliest_arrival_) {
        ++stats_.pruned_by_earliest_arrival_;
        continue;
      }
      for (auto stop_idx = entry.to_stop_index_ - 1;
           stop_idx >= entry.from_stop_index_; --stop_idx) {
        for (auto const& transfer :
             data_.reverse_transfers_.at(entry.trip_, stop_idx)) {
          ++stats_.transfers_scanned_;
          enqueue(transfer.from_trip_, transfer.from_stop_idx_, transfers + 1,
                  current_trip_segment);
        }
        if (stop_idx == 0) {
          break;
        }
      }
    }
  }

  inline void enqueue(trip_id trip, stop_idx_t stop_index, unsigned transfers,
                      std::size_t previous_trip_segment) {
    assert(transfers < queues_.size());
    if (Dir == search_dir::FWD) {
      auto const old_first_reachable = first_reachable_stop_[trip];
      if (stop_index >= old_first_reachable) {
        return;
      }
      auto& queue = queues_[transfers];  // NOLINT
      queue.emplace_back(trip, stop_index, old_first_reachable,
                         previous_trip_segment);

      auto const line = data_.trip_to_line_[trip];
      for (trip_id t = trip;
           t < data_.trip_idx_end_ && data_.trip_to_line_[t] == line; ++t) {
        first_reachable_stop_[t] =
            std::min(first_reachable_stop_[t], stop_index);
      }
    } else {
      auto const old_last_reachable = first_reachable_stop_[trip];
      if (stop_index <= old_last_reachable) {
        return;
      }
      auto& queue = queues_[transfers];  // NOLINT
      queue.emplace_back(trip, old_last_reachable, stop_index,
                         previous_trip_segment);
      auto const line = data_.trip_to_line_[trip];
      for (trip_id t = trip; data_.trip_to_line_[t] == line; --t) {
        first_reachable_stop_[t] =
            std::max(first_reachable_stop_[t], stop_index);
        if (t == 0) {
          break;
        }
      }
    }
  }

  void destination_reached(time arrival_time, unsigned transfers,
                           destination_arrival const& dest_arrival,
                           std::size_t queue_entry) {
    ++stats_.destination_reached_;

    auto const travel_time = arrival_time > start_time
                                 ? arrival_time - start_time
                                 : start_time - arrival_time;

    if (travel_time > MAX_TRAVEL_TIME) {
      ++stats_.max_travel_time_reached_;
      return;
    }

    if (Dir == search_dir::FWD) {
      auto const dest_index = dest_arrival.footpath_.to_stop_;
      auto const previous_earliest_arrival = earliest_arrival_[dest_index];
      if (arrival_time < previous_earliest_arrival) {
        earliest_arrival_[dest_index] = arrival_time;
        if (total_earliest_arrival_ == previous_earliest_arrival) {
          total_earliest_arrival_ = get_total_earliest_arrival();
        }
        add_result(
            journeys_[dest_index],
            {Dir, start_time, arrival_time, transfers, transfers + 1,
             dest_arrival.footpath_.to_stop_, &dest_arrival, queue_entry});
      }
    } else {
      auto const dest_index = dest_arrival.footpath_.from_stop_;
      auto const previous_latest_departure = earliest_arrival_[dest_index];
      if (arrival_time > previous_latest_departure) {
        earliest_arrival_[dest_index] = arrival_time;
        if (total_earliest_arrival_ == previous_latest_departure) {
          total_earliest_arrival_ = get_total_earliest_arrival();
        }
        add_result(
            journeys_[dest_index],
            {Dir, start_time, arrival_time, transfers, transfers + 1,
             dest_arrival.footpath_.from_stop_, &dest_arrival, queue_entry});
      }
    }
  }

  time get_total_earliest_arrival() const {
    auto const latest =
        (Dir == search_dir::FWD &&
         destination_mode_ == destination_mode::ALL) ||
        (Dir == search_dir::BWD && destination_mode_ == destination_mode::ANY);
    time result = latest ? std::numeric_limits<time>::min()
                         : std::numeric_limits<time>::max();
    for (auto const& station : destination_stations_) {
      if (latest) {
        result = std::max(result, earliest_arrival_[station]);
      } else {
        result = std::min(result, earliest_arrival_[station]);
      }
    }
    return result;
  }

  void add_result(std::vector<tb_journey>& journeys, tb_journey new_result) {
    if (std::any_of(begin(journeys), end(journeys), [&](auto const& existing) {
          return existing.dominates(new_result);
        })) {
      return;
    }

    utl::erase_if(journeys, [&](auto const& existing) {
      return new_result.dominates(existing);
    });

    journeys.push_back(new_result);
    ++stats_.results_added_;
  }

  inline bool is_start(station_id station) const {
    return std::find(begin(start_stations_), end(start_stations_), station) !=
           end(start_stations_);
  }

  inline duration get_initial_duration(station_id station) {
    return static_cast<duration>(Dir == search_dir::FWD
                                     ? start_times_[station] - start_time
                                     : start_time - start_times_[station]);
  }

  tb_data const& data_;
  schedule const& sched_;
  time const start_time;
  bool count_initial_transfer_time_;
  bool count_final_transfer_time_;
  destination_mode destination_mode_;
  std::vector<station_id> start_stations_;
  std::vector<station_id> destination_stations_;
  std::map<station_id, time> start_times_;
  std::vector<std::vector<destination_arrival>> destination_arrivals_;
  std::vector<std::vector<tb_journey>> journeys_;
  std::vector<time> earliest_arrival_;
  time total_earliest_arrival_{INVALID};
  std::array<std::vector<queue_entry>, MAX_TRANSFERS + 1> queues_;
  std::vector<stop_idx_t> first_reachable_stop_;
  tb_statistics stats_{};
};

}  // namespace motis::tripbased
