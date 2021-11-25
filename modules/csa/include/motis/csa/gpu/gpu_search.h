#pragma once

#include <cinttypes>
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/core/common/raii.h"
#include "motis/csa/collect_start_times.h"
#include "motis/csa/csa_reconstruction.h"

#include "gpu_csa.h"

namespace motis::csa {

template <typename T>
struct arrivals_wrapper {
  arrivals_wrapper(T* ptr, uint32_t const size, uint32_t const query_idx)
      : ptr_{ptr}, size_{size}, query_idx_{query_idx} {}

  T* operator[](size_t const element_index) const {
    return ptr_ + (query_idx_ * size_ * (GPU_CSA_MAX_TRANSFERS + 1)) +
           element_index * (GPU_CSA_MAX_TRANSFERS + 1);
  }

  size_t size() const { return size_; }

  T* ptr_;
  uint32_t size_, query_idx_;
};

template <typename BlockType>
struct trip_reachable_wrapper {
  trip_reachable_wrapper(BlockType const* blocks, uint32_t const size,
                         uint32_t const query_idx)
      : blocks_{blocks}, size_{size}, query_idx_{query_idx} {}

  struct single_trip_wrapper {
    bool operator[](uint32_t const transfers) const {
      return blocks_[offset_ + transfers] !=
             std::numeric_limits<gpu_csa_con_idx>::max();
    }
    BlockType const* blocks_;
    uint32_t offset_;
  };

  single_trip_wrapper operator[](uint32_t const trip_idx) const {
    return {blocks_, (query_idx_ * size_ * (GPU_CSA_MAX_TRANSFERS + 1)) +
                         trip_idx * (GPU_CSA_MAX_TRANSFERS + 1)};
  }

  uint32_t size() const { return size_; }

  BlockType const* blocks_;
  uint32_t size_;
  uint32_t query_idx_;
};

template <typename BlockType>
trip_reachable_wrapper(BlockType*, uint32_t, uint32_t)
    -> trip_reachable_wrapper<BlockType>;

struct gpu_search {
  static constexpr time INVALID = std::numeric_limits<time>::max();

  gpu_search(schedule const& sched, csa_timetable const& tt, csa_query const& q,
             csa_statistics& stats)
      : sched_{sched}, tt_{tt}, q_{q}, stats_{stats} {}

  template <typename Results>
  void search_in_interval(Results& results, interval const& search_interval,
                          bool const ontrip_at_interval_end) {
    // Setup query.
    search(results, collect_start_times(tt_, q_, search_interval,
                                        ontrip_at_interval_end));
  }

  template <typename Results>
  void search(Results& results, std::set<motis::time> const& start_times) {
    if (start_times.empty()) {
      return;
    }

    std::map<station_id, time> start_offsets;
    for (auto const& start_station_idx : q_.meta_starts_) {
      for (auto const& fp : tt_.stations_[start_station_idx].footpaths_) {
        auto const start_offset = static_cast<duration>(
            fp.from_station_ == fp.to_station_ ? 0U : fp.duration_);
        auto const [it, inserted] =
            start_offsets.emplace(fp.to_station_, start_offset);
        if (!inserted) {
          it->second = std::min(it->second, start_offset);
        }
      }
    }
    auto const get_meta_start_times = [&](time const start_time) {
      auto meta_start_times = std::map<station_id, time>{};
      for (auto const& s : q_.meta_starts_) {
        meta_start_times.emplace(s, start_offsets[s] + start_time);
      }
      return meta_start_times;
    };

    auto time_limit = motis::time{0U};
    std::vector<gpu_csa_start> starts;
    for (auto const& [query_idx, start_time] : utl::enumerate(start_times)) {
      for (auto const& [start_station_idx, offset] : start_offsets) {
        starts.emplace_back(
            gpu_csa_start{static_cast<uint32_t>(query_idx), start_station_idx,
                          static_cast<gpu_csa_time>(start_time + offset)});
        time_limit = std::max(
            time_limit,
            static_cast<motis::time>(start_time + offset + MAX_TRAVEL_TIME));
      }
    }

    // Compute start bucket.
    auto const start_bucket = static_cast<uint32_t>(std::distance(
        begin(tt_.fwd_bucket_starts_),
        std::lower_bound(begin(tt_.fwd_bucket_starts_),
                         end(tt_.fwd_bucket_starts_), *begin(start_times),
                         [&](uint32_t const& bucket_idx, motis::time const& t) {
                           return tt_.fwd_connections_[bucket_idx].departure_ <
                                  t;
                         })));

    // Compute response.
    MOTIS_START_TIMING(search_timing);
    auto res = ::gpu_csa_search(tt_.gpu_timetable_.ptr_, starts.data(),
                                static_cast<uint32_t>(starts.size()),
                                static_cast<uint32_t>(start_times.size()),
                                start_bucket, time_limit);
    MOTIS_STOP_TIMING(search_timing);
    if (res.station_arrivals_ == nullptr || res.trip_reachable_ == nullptr) {
      throw std::runtime_error{"error in GPU code"};
    }
    MOTIS_FINALLY([&]() { gpu_csa_free_result(&res); });

    // Extract optimal journeys.
    MOTIS_START_TIMING(reconstruction_timing);
    uint32_t q_idx = 0U;
    for (auto const& start_time : start_times) {
      auto station_arrivals =
          arrivals_wrapper{res.station_arrivals_,
                           static_cast<uint32_t>(tt_.stations_.size()), q_idx};
      auto trip_reachable = trip_reachable_wrapper{
          res.trip_reachable_, static_cast<uint32_t>(tt_.trip_count_), q_idx};

      for (auto const& target_station_idx : q_.meta_dests_) {
        auto const& station_arrival = station_arrivals[target_station_idx];
        for (auto transfers = 0U; transfers <= MAX_TRANSFERS; ++transfers) {
          if (station_arrival[transfers] != INVALID) {
            csa_journey j{search_dir::FWD, start_time,
                          station_arrival[transfers], transfers,
                          &tt_.stations_[target_station_idx]};
            csa_reconstruction<search_dir::FWD, decltype(station_arrivals),
                               decltype(trip_reachable)>{
                tt_, get_meta_start_times(start_time), station_arrivals,
                trip_reachable}
                .extract_journey(j);
            if (j.duration() <= GPU_CSA_MAX_TRAVEL_TIME) {
              results.push_back(j);
            }
          }
        }
      }
      ++q_idx;
    }
    MOTIS_STOP_TIMING(reconstruction_timing);

    stats_.search_duration_ += MOTIS_TIMING_MS(search_timing);
    stats_.reconstruction_duration_ += MOTIS_TIMING_MS(reconstruction_timing);
  }

  schedule const& sched_;
  csa_timetable const& tt_;
  csa_query const& q_;
  csa_statistics& stats_;
};

}  // namespace motis::csa