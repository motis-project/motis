#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "utl/erase_if.h"
#include "utl/get_or_create.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/edges.h"

#include "motis/tripbased/data.h"
#include "motis/tripbased/tb_journey.h"
#include "motis/tripbased/tb_search_common.h"
#include "motis/tripbased/tb_statistics.h"

#include "motis/tripbased/limits.h"

namespace motis::tripbased {

// https://stackoverflow.com/questions/49318316/initialize-all-elements-or-stdarray-with-the-same-constructor-arguments
template <typename T, std::size_t N, std::size_t Idx = N>
struct array_maker {
  template <typename... Ts>
  static std::array<T, N> make_array(const T& v, Ts... tail) {
    return array_maker<T, N, Idx - 1>::make_array(v, v, tail...);
  }
};

template <typename T, std::size_t N>
struct array_maker<T, N, 1> {
  template <typename... Ts>
  static std::array<T, N> make_array(const T& v, Ts... tail) {
    return std::array<T, N>{v, tail...};
  }
};

template <search_dir Dir = search_dir::FWD>
struct tb_profile_search {
  static constexpr time INVALID = Dir == search_dir::FWD
                                      ? std::numeric_limits<time>::max()
                                      : std::numeric_limits<time>::min();

  tb_profile_search(tb_data const& data, schedule const& sched,
                    time interval_begin, time interval_end,
                    bool count_initial_transfer_time,
                    bool count_final_transfer_time, destination_mode dest_mode)
      : data_(data),
        sched_(sched),
        interval_begin_(interval_begin),
        interval_end_(interval_end),
        count_initial_transfer_time_(count_initial_transfer_time),
        count_final_transfer_time_(count_final_transfer_time),
        destination_mode_(dest_mode),
        destination_arrivals_(data.line_count_),
        total_earliest_arrival_(
            array_maker<time, MAX_TRANSFERS + 1>::make_array(INVALID)),
        first_reachable_stop_(
            data.trip_count_,
            array_maker<stop_idx_t, MAX_TRANSFERS + 1>::make_array(
                Dir == search_dir::FWD
                    ? std::numeric_limits<stop_idx_t>::max()
                    : std::numeric_limits<stop_idx_t>::min())) {}

  void add_start(station_id stop_id, time initial_duration,
                 bool allow_footpaths = true) {
    assert(stop_id < sched_.stations_.size());
    start_stations_.emplace_back(stop_id, initial_duration, allow_footpaths);
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
    earliest_arrival_.resize(
        sched_.stations_.size(),
        array_maker<time, MAX_TRANSFERS + 1>::make_array(INVALID));
    journeys_.resize(sched_.stations_.size());
    if (Dir == search_dir::FWD) {
      for (start_time_ = static_cast<time>(interval_end_ + 1);
           start_time_ >= interval_begin_;
           start_time_ = next_iteration_start_time_) {
        search_iteration();
      }
    } else {
      for (start_time_ = static_cast<time>(interval_begin_ - 1);
           start_time_ <= interval_end_;
           start_time_ = next_iteration_start_time_) {
        search_iteration();
      }
    }
  }

  std::vector<tb_journey>& get_results(station_id destination) {
    return results_.at(destination);
  }

  tb_statistics& get_statistics() { return stats_; }
  tb_statistics const& get_statistics() const { return stats_; }

  std::pair<time, time> get_interval() const {
    return {interval_begin_, interval_end_};
  }

private:
  void add_starts() {
    next_iteration_start_time_ = INVALID_TIME;
    for (auto const& [stop_id, initial_duration, allow_footpaths] :
         start_stations_) {
      auto const station_arrival = Dir == search_dir::FWD
                                       ? start_time_ + initial_duration
                                       : start_time_ - initial_duration;
      start_times_[stop_id] = station_arrival;
      add_start({stop_id, stop_id,
                 count_initial_transfer_time_
                     ? static_cast<unsigned>(
                           sched_.stations_[stop_id]->transfer_time_)
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
    }
    if (Dir == search_dir::FWD) {
      auto const next = static_cast<time>(start_time_ - 1);
      if (next_iteration_start_time_ == INVALID_TIME ||
          next_iteration_start_time_ > next) {
        next_iteration_start_time_ = next;
      }
    } else {
      auto const next = static_cast<time>(start_time_ + 1);
      if (next_iteration_start_time_ == INVALID_TIME ||
          next_iteration_start_time_ < next) {
        next_iteration_start_time_ = next;
      }
    }
  }

  void add_start(tb_footpath const& fp, time initial_duration) {
    auto const offset = initial_duration + fp.duration_;
    auto const arrival_time = static_cast<time>(
        Dir == search_dir::FWD ? start_time_ + offset : start_time_ - offset);
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
      auto const [trip, next_trip] =
          Dir == search_dir::FWD ? data_.first_and_previous_reachable_trip(
                                       line, stop_idx, arrival_time)
                                 : data_.last_and_next_reachable_trip(
                                       line, stop_idx, arrival_time);
      if (trip) {
        enqueue(trip->first, stop_idx, 0, 0);
      }
      if (next_trip) {
        update_next_iteration_start_time(next_trip->second, offset);
      }
    }
  }

  void update_next_iteration_start_time(time t, time offset) {
    if (Dir == search_dir::FWD) {
      auto const new_time = t - offset;
      if (new_time >= start_time_) {
        return;
      }
      if (next_iteration_start_time_ == INVALID_TIME ||
          next_iteration_start_time_ < new_time) {
        next_iteration_start_time_ = new_time;
      }
    } else {
      auto const new_time = t + offset;
      if (new_time <= start_time_) {
        return;
      }
      if (next_iteration_start_time_ == INVALID_TIME ||
          next_iteration_start_time_ > new_time) {
        next_iteration_start_time_ = new_time;
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

  void search_iteration() {
    ++stats_.search_iterations_;
    add_starts();

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
      auto const size =
          std::max(static_cast<uint64_t>(queues_[i].size()),  // NOLINT
                   stats_.queue_size_[i]);  // NOLINT
      stats_.queue_size_[i] = size;  // NOLINT
      stats_.max_queue_size_ = std::max(stats_.max_queue_size_, size);
    }

    add_iteration_results();
    prepare_next_iteration();
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

      assert(transfers < MAX_TRANSFERS);

      auto const next_stop_arrival =
          data_.arrival_times_[entry.trip_][entry.from_stop_index_ + 1];
      if (next_stop_arrival >=
          total_earliest_arrival_[transfers + 1]) {  // NOLINT
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

      assert(transfers < MAX_TRANSFERS);

      assert(entry.to_stop_index_ > 0);
      auto const prev_stop_departure =
          data_.departure_times_[entry.trip_][entry.to_stop_index_ - 1];
      if (prev_stop_departure <=
          total_earliest_arrival_[transfers + 1]) {  // NOLINT
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
      auto const old_first_reachable =
          first_reachable_stop_[trip][transfers];  // NOLINT
      if (stop_index >= old_first_reachable) {
        return;
      }
      auto& queue = queues_[transfers];  // NOLINT
      queue.emplace_back(trip, stop_index, old_first_reachable,
                         previous_trip_segment);

      auto const line = data_.trip_to_line_[trip];
      for (trip_id t = trip;
           t < data_.trip_count_ && data_.trip_to_line_[t] == line; ++t) {
        for (auto trfs = transfers; trfs <= MAX_TRANSFERS; ++trfs) {
          first_reachable_stop_[t][trfs] =
              std::min(first_reachable_stop_[t][trfs], stop_index);
        }
      }
    } else {
      auto const old_last_reachable = first_reachable_stop_[trip][transfers];
      if (stop_index <= old_last_reachable) {
        return;
      }
      auto& queue = queues_[transfers];  // NOLINT
      queue.emplace_back(trip, old_last_reachable, stop_index,
                         previous_trip_segment);
      auto const line = data_.trip_to_line_[trip];
      for (trip_id t = trip; data_.trip_to_line_[t] == line; --t) {
        for (auto trfs = transfers; trfs <= MAX_TRANSFERS; ++trfs) {
          first_reachable_stop_[t][trfs] =
              std::max(first_reachable_stop_[t][trfs], stop_index);
        }
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

    auto const travel_time = arrival_time > start_time_
                                 ? arrival_time - start_time_
                                 : start_time_ - arrival_time;

    if (travel_time > MAX_TRAVEL_TIME) {
      ++stats_.max_travel_time_reached_;
      return;
    }

    if (Dir == search_dir::FWD) {
      auto const dest_index = dest_arrival.footpath_.to_stop_;
      for (auto trfs = transfers; trfs <= MAX_TRANSFERS; ++trfs) {
        auto const previous_earliest_arrival =
            earliest_arrival_[dest_index][trfs];  // NOLINT
        if (arrival_time >= previous_earliest_arrival) {
          continue;
        }
        earliest_arrival_[dest_index][trfs] = arrival_time;  // NOLINT
        if (total_earliest_arrival_[trfs] ==  // NOLINT
            previous_earliest_arrival) {
          total_earliest_arrival_[trfs] =  // NOLINT
              get_total_earliest_arrival(trfs);
          ++stats_.total_earliest_arrival_updates_;
          if (total_earliest_arrival_[trfs] != INVALID &&  // NOLINT
              stats_.all_destinations_reached_ == 0) {
            stats_.all_destinations_reached_ = stats_.destination_reached_;
          }
        }
        if (trfs == transfers) {
          add_result(
              journeys_[dest_index],
              {Dir, start_time_, arrival_time, transfers, transfers + 1,
               dest_arrival.footpath_.to_stop_, &dest_arrival, queue_entry});
        }
      }
    } else {
      auto const dest_index = dest_arrival.footpath_.from_stop_;
      for (auto trfs = transfers; trfs <= MAX_TRANSFERS; ++trfs) {
        auto const previous_latest_departure =
            earliest_arrival_[dest_index][trfs];  // NOLINT
        if (arrival_time <= previous_latest_departure) {
          continue;
        }
        earliest_arrival_[dest_index][trfs] = arrival_time;  // NOLINT
        if (total_earliest_arrival_[trfs] ==  // NOLINT
            previous_latest_departure) {
          total_earliest_arrival_[trfs] =  // NOLINT
              get_total_earliest_arrival(trfs);
          ++stats_.total_earliest_arrival_updates_;
          if (total_earliest_arrival_[trfs] != INVALID &&  // NOLINT
              stats_.all_destinations_reached_ == 0) {
            stats_.all_destinations_reached_ = stats_.destination_reached_;
          }
        }
        if (trfs == transfers) {
          add_result(
              journeys_[dest_index],
              {Dir, start_time_, arrival_time, transfers, transfers + 1,
               dest_arrival.footpath_.from_stop_, &dest_arrival, queue_entry});
        }
      }
    }
  }

  time get_total_earliest_arrival(unsigned const trfs) const {
    auto const latest =
        (Dir == search_dir::FWD &&
         destination_mode_ == destination_mode::ALL) ||
        (Dir == search_dir::BWD && destination_mode_ == destination_mode::ANY);
    time result = latest ? std::numeric_limits<time>::min()
                         : std::numeric_limits<time>::max();
    for (auto const& station : destination_stations_) {
      if (latest) {
        result = std::max(result, earliest_arrival_[station][trfs]);  // NOLINT
      } else {
        result = std::min(result, earliest_arrival_[station][trfs]);  // NOLINT
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

  void reconstruct_journey(tb_journey& j) {
    reconstruct_tb_journey<Dir>(
        j, data_, sched_, queues_, start_times_, count_initial_transfer_time_,
        [this](station_id station) { return is_start(station); },
        [this](station_id station) { return get_initial_duration(station); },
        stats_);
  }

  inline bool is_start(station_id station) const {
    return std::find_if(begin(start_stations_), end(start_stations_),
                        [station](auto const& s) {
                          return std::get<0>(s) == station;
                        }) != end(start_stations_);
  }

  inline duration get_initial_duration(station_id station) {
    return static_cast<duration>(Dir == search_dir::FWD
                                     ? start_times_[station] - start_time_
                                     : start_time_ - start_times_[station]);
  }

  void add_iteration_results() {
    for (auto const destination : destination_stations_) {
      auto& journeys = journeys_[destination];
      auto& results = utl::get_or_create(
          results_, destination, []() { return std::vector<tb_journey>(); });
      results.reserve(results.size() + journeys.size());
      for (auto& j : journeys) {
        reconstruct_journey(j);
        results.emplace_back(std::move(j));
        ++result_count_;
      }
      journeys.clear();
    }
  }

  void prepare_next_iteration() {
    for (auto& q : queues_) {
      q.clear();
    }
  }

  tb_data const& data_;
  schedule const& sched_;
  time interval_begin_, interval_end_;
  bool count_initial_transfer_time_;
  bool count_final_transfer_time_;
  time start_time_{INVALID_TIME};
  time next_iteration_start_time_{INVALID_TIME};
  destination_mode destination_mode_;
  std::vector<std::tuple<station_id, time, bool>> start_stations_;
  std::vector<station_id> destination_stations_;
  std::map<station_id, time> start_times_;
  std::vector<std::vector<destination_arrival>> destination_arrivals_;
  std::vector<std::vector<tb_journey>> journeys_;
  std::map<station_id, std::vector<tb_journey>> results_;
  unsigned result_count_{0};
  std::vector<std::array<time, MAX_TRANSFERS + 1>> earliest_arrival_;
  std::array<time, MAX_TRANSFERS + 1> total_earliest_arrival_;
  std::array<std::vector<queue_entry>, MAX_TRANSFERS + 1> queues_;
  std::vector<std::array<stop_idx_t, MAX_TRANSFERS + 1>> first_reachable_stop_;
  tb_statistics stats_{};
};

}  // namespace motis::tripbased
