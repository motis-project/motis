#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "motis/vector.h"

#include "motis/core/common/fws_multimap.h"
#include "motis/core/schedule/footpath.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

namespace motis::tripbased {

using trip_id = uint32_t;
using station_id = uint32_t;
using line_id = uint32_t;
using stop_idx_t = uint16_t;

struct tb_footpath {
  tb_footpath() = default;
  tb_footpath(station_id from_stop, station_id to_stop, unsigned duration)
      : from_stop_(from_stop), to_stop_(to_stop), duration_(duration) {}
  explicit tb_footpath(footpath const& fp)
      : from_stop_(fp.from_station_),
        to_stop_(fp.to_station_),
        duration_(fp.duration_) {}

  bool is_interstation_walk() const { return from_stop_ != to_stop_; }

  station_id from_stop_{};
  station_id to_stop_{};
  uint32_t duration_{};
};

struct line_stop {
  line_stop() = default;
  line_stop(line_id line, stop_idx_t stop_idx)
      : line_(line), stop_idx_(stop_idx) {}

  line_id line_{};
  stop_idx_t stop_idx_{};
  uint8_t padding_[1]{};
};

struct tb_transfer {
  tb_transfer() = default;
  tb_transfer(trip_id to_trip, stop_idx_t to_stop_idx)
      : to_trip_(to_trip), to_stop_idx_(to_stop_idx) {}

  bool valid() const {
    return to_stop_idx_ != std::numeric_limits<stop_idx_t>::max();
  }

  trip_id to_trip_{};
  stop_idx_t to_stop_idx_{std::numeric_limits<stop_idx_t>::max()};
  uint8_t padding_[1]{};
};

struct tb_reverse_transfer {
  tb_reverse_transfer() = default;
  tb_reverse_transfer(trip_id from_trip, stop_idx_t from_stop_idx,
                      stop_idx_t to_stop_idx)
      : from_trip_(from_trip),
        from_stop_idx_(from_stop_idx),
        to_stop_idx_(to_stop_idx) {}

  bool valid() const {
    return from_stop_idx_ != std::numeric_limits<stop_idx_t>::max();
  }

  trip_id from_trip_{};
  stop_idx_t from_stop_idx_{std::numeric_limits<stop_idx_t>::max()};
  stop_idx_t to_stop_idx_{std::numeric_limits<stop_idx_t>::max()};
};

struct tb_data {
  std::optional<std::pair<trip_id, time>> first_reachable_trip(
      line_id line, stop_idx_t stop_idx, time earliest_departure) const {
    assert(line < line_count_);
    for (auto trip = line_to_first_trip_[line];
         trip < trip_count_ && trip_to_line_[trip] == line; ++trip) {
      auto const dep_time = departure_times_[trip][stop_idx];
      if (dep_time >= earliest_departure) {
        return std::make_pair(trip, dep_time);
      }
    }
    return {};
  }

  std::pair<std::optional<std::pair<trip_id, time>>,
            std::optional<std::pair<trip_id, time>>>
  first_and_previous_reachable_trip(line_id line, stop_idx_t stop_idx,
                                    time earliest_departure) const {
    assert(line < line_count_);
    auto const first_trip_in_line = line_to_first_trip_[line];
    auto const last_trip_in_line = line_to_last_trip_[line];
    for (auto trip = first_trip_in_line; trip <= last_trip_in_line; ++trip) {
      auto const dep_time = departure_times_[trip][stop_idx];
      if (dep_time >= earliest_departure) {
        if (trip != first_trip_in_line) {
          return {{{trip, dep_time}},
                  {{trip - 1, departure_times_[trip - 1][stop_idx]}}};
        } else {
          return {{{trip, dep_time}}, {}};
        }
      }
    }
    return {
        {},
        {{last_trip_in_line, departure_times_[last_trip_in_line][stop_idx]}}};
  }

  std::optional<std::pair<trip_id, time>> last_reachable_trip(
      line_id line, stop_idx_t stop_idx, time latest_arrival) const {
    assert(line < line_count_);
    for (auto trip = static_cast<int>(line_to_last_trip_[line]);
         trip >= 0 && trip_to_line_[trip] == line; --trip) {
      auto const arr_time = arrival_times_[trip][stop_idx];
      if (arr_time <= latest_arrival) {
        return std::make_pair(trip, arr_time);
      }
    }
    return {};
  }

  std::pair<std::optional<std::pair<trip_id, time>>,
            std::optional<std::pair<trip_id, time>>>
  last_and_next_reachable_trip(line_id line, stop_idx_t stop_idx,
                               time latest_arrival) const {
    assert(line < line_count_);
    auto const first_trip_in_line = line_to_first_trip_[line];
    auto const last_trip_in_line = line_to_last_trip_[line];
    for (auto trip = static_cast<int>(line_to_last_trip_[line]);
         trip >= static_cast<int>(first_trip_in_line); --trip) {
      auto const arr_time = arrival_times_[trip][stop_idx];
      if (arr_time <= latest_arrival) {
        if (trip != last_trip_in_line) {
          return {{{trip, arr_time}},
                  {{trip + 1, arrival_times_[trip + 1][stop_idx]}}};
        } else {
          return {{{trip, arr_time}}, {}};
        }
      }
    }
    return {
        {},
        {{first_trip_in_line, arrival_times_[first_trip_in_line][stop_idx]}}};
  }

  uint64_t trip_count_{};
  uint64_t line_count_{};

  mcd::vector<trip_id> line_to_first_trip_;
  mcd::vector<trip_id> line_to_last_trip_;
  mcd::vector<line_id> trip_to_line_;
  mcd::vector<stop_idx_t> line_stop_count_;

  fws_multimap<tb_footpath, station_id> footpaths_{};
  fws_multimap<tb_footpath, station_id> reverse_footpaths_{};
  fws_multimap<line_stop, station_id> lines_at_stop_{};
  fws_multimap<station_id, line_id> stops_on_line_{};

  fws_multimap<motis::time> arrival_times_{};
  shared_idx_fws_multimap<motis::time> departure_times_{arrival_times_.index_};
  nested_fws_multimap<tb_transfer> transfers_{arrival_times_.index_};
  nested_fws_multimap<tb_reverse_transfer> reverse_transfers_{
      arrival_times_.index_};

  shared_idx_fws_multimap<uint8_t, line_id> in_allowed_{stops_on_line_.index_};
  shared_idx_fws_multimap<uint8_t, line_id> out_allowed_{stops_on_line_.index_};
  shared_idx_fws_multimap<uint16_t, line_id> arrival_platform_{
      stops_on_line_.index_};
  shared_idx_fws_multimap<uint16_t, line_id> departure_platform_{
      stops_on_line_.index_};
};

}  // namespace motis::tripbased
