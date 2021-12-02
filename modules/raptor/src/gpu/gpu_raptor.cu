#include "motis/raptor/gpu/gpu_raptor.cuh"

#include "cooperative_groups.h"
#include "cuda_profiler_api.h"

namespace motis::raptor {

using namespace cooperative_groups;

// leader type must be unsigned 32bit
// no leader is a zero ballot vote (all 0) minus 1 => with underflow all 1's
constexpr unsigned int FULL_MASK = 0xFFFFffff;
constexpr unsigned int NO_LEADER = FULL_MASK;

__device__ __forceinline__ unsigned int get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned int get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned int get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned int get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

__device__ void mark(unsigned int* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const mask = 1 << (idx % 32);
  atomicOr(&store[store_idx], mask);
}

__device__ bool marked(unsigned int const* const store, unsigned int idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const val = store[store_idx];
  unsigned int const mask = 1 << (idx % 32);
  return (bool)(val & mask);
}

__device__ void reset_store(unsigned int* store, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();

  for (auto idx = t_id; idx < store_size; idx += stride) {
    store[idx] = 0;
  }
}

__device__ void convert_station_to_route_marks(unsigned int* station_marks,
                                               unsigned int* route_marks,
                                               bool* any_station_marked,
                                               device_gpu_timetable const& tt) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for (auto idx = global_t_id; idx < tt.stop_count_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!*any_station_marked) {
        *any_station_marked = true;
      }
      auto const stop = tt.stops_[idx];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        mark(route_marks, tt.stop_routes_[sri]);
      }
    }
  }
}

__device__ bool update_arrival(time* const base, stop_id const s_id,
                               time const val) {

#if __CUDA_ARCH__ >= 700

  auto old_value = base[s_id];
  time assumed;

  do {
    if (old_value <= val) {
      return false;
    }

    assumed = old_value;

    old_value = atomicCAS(&base[s_id], assumed, val);
  } while (assumed != old_value);

  return true;

#else

  // we have a 16-bit time value array, but only 32-bit atomic operations
  // therefore every two 16-bit time values are read as one 32-bit time value
  // then they are the corresponding part is updated and stored if a better
  // time value was found while the remaining 16 bit value part remains
  // unchanged

  time* const arr_address = &base[s_id];
  unsigned int* base_address = (unsigned int*)((size_t)arr_address & ~2);
  unsigned int old_value, assumed, new_value, compare_val;

  old_value = *base_address;

  do {
    assumed = old_value;

    if ((size_t)arr_address & 2) {
      compare_val = (0x0000FFFF & assumed) ^ (((unsigned int)val) << 16);
    } else {
      compare_val = (0xFFFF0000 & assumed) ^ (unsigned int)val;
    }

    new_value = __vminu2(old_value, compare_val);

    if (new_value == old_value) {
      return false;
    }

    old_value = atomicCAS(base_address, assumed, new_value);
  } while (assumed != old_value);

  return true;

#endif
}

__device__ void copy_marked_arrivals(time* const to, time const* const from,
                                     unsigned int* station_marks,
                                     device_gpu_timetable const& tt) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  for (; arr_idx < tt.stop_count_; arr_idx += global_stride) {
    if (marked(station_marks, arr_idx)) {
      to[arr_idx] = from[arr_idx];
    } else {
      to[arr_idx] = invalid<time>;
    }
  }
}

__device__ void copy_and_min_arrivals(time* const to, time* const from,
                                      device_gpu_timetable const& tt) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  for (; arr_idx < tt.stop_count_; arr_idx += global_stride) {
    to[arr_idx] = min(from[arr_idx], to[arr_idx]);
  }
}

__device__ void update_route_larger32(gpu_route const& route,
                                      time const* const prev_arrivals,
                                      time* const arrivals,
                                      unsigned int* station_marks,
                                      device_gpu_timetable const& tt) {
  auto const t_id = threadIdx.x;

  stop_id stop_id_t = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;

  int active_stop_count = route.stop_count_;

  // this is ceil(stop_count / 32)
  int const stage_count = (route.stop_count_ + (32 - 1)) >> 5;
  int active_stage_count = stage_count;

  unsigned int leader = NO_LEADER;
  unsigned int any_arrival = 0;

  for (int trip_offset = 0; trip_offset < route.trip_count_; ++trip_offset) {

    for (int current_stage = 0; current_stage < active_stage_count;
         ++current_stage) {

      int stage_id = (current_stage << 5) + t_id;

      // load the prev arrivals for the current stage
      if (stage_id < active_stop_count) {
        stop_id_t = tt.route_stops_[route.index_to_route_stops_ + stage_id];
        //        prev_arrival = get_arrival(prev_arrivals, stop_id_t);
        prev_arrival = prev_arrivals[stop_id_t];
      }

      any_arrival |= __any_sync(FULL_MASK, valid(prev_arrival));
      if (current_stage == active_stage_count - 1 && !any_arrival) {
        return;
      }

      if (!any_arrival) {
        continue;
      }

      // load the stop times for the current stage
      if (stage_id < active_stop_count) {
        auto const st_idx = route.index_to_stop_times_ +
                            (trip_offset * route.stop_count_) + stage_id;
        stop_departure = tt.stop_departures_[st_idx];
      }

      // get the current stage leader
      unsigned int ballot = __ballot_sync(
          FULL_MASK, (stage_id < active_stop_count) && valid(prev_arrival) &&
                         valid(stop_departure) &&
                         (prev_arrival <= stop_departure));
      leader = __ffs(ballot) - 1;

      if (leader != NO_LEADER) {
        leader += current_stage << 5;
      }

      // first update the current stage
      if (leader != NO_LEADER && stage_id < active_stop_count) {

        if (stage_id > leader) {
          auto const st_idx = route.index_to_stop_times_ +
                              (trip_offset * route.stop_count_) + stage_id;
          stop_arrival = tt.stop_arrivals_[st_idx];
          bool updated = update_arrival(arrivals, stop_id_t, stop_arrival);
          if (updated) {
            mark(station_marks, stop_id_t);
          }
        }
      }

      // then update all upward stages
      if (leader != NO_LEADER) {
        for (int upward_stage = current_stage + 1;
             upward_stage < active_stage_count; ++upward_stage) {

          int upwards_id = (upward_stage << 5) + t_id;
          if (upwards_id < active_stop_count) {

            auto const st_idx = route.index_to_stop_times_ +
                                (trip_offset * route.stop_count_) + upwards_id;

            stop_arrival = tt.stop_arrivals_[st_idx];
            stop_id_t =
                tt.route_stops_[route.index_to_route_stops_ + upwards_id];
            bool updated = update_arrival(arrivals, stop_id_t, stop_arrival);
            if (updated) {
              mark(station_marks, stop_id_t);
            }
          }
        }

        // for this route we do not need to update any station higher than the
        // leader anymore
        active_stop_count = leader;
        active_stage_count = (active_stop_count + (32 - 1)) >> 5;
        leader = NO_LEADER;
      }
    }
  }
}

__device__ void update_route_smaller32(gpu_route const route,
                                       time const* const prev_arrivals,
                                       time* const arrivals,
                                       unsigned int* station_marks,
                                       device_gpu_timetable const& tt) {
  auto const t_id = threadIdx.x;

  stop_id stop_id_t = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;

  unsigned leader = route.stop_count_;
  unsigned int active_stop_count = route.stop_count_;

  if (t_id < active_stop_count) {
    stop_id_t = tt.route_stops_[route.index_to_route_stops_ + t_id];
    //    prev_arrival = get_arrival(prev_arrivals, stop_id_t);
    prev_arrival = prev_arrivals[stop_id_t];
  }

  if (!__any_sync(FULL_MASK, valid(prev_arrival))) {
    return;
  }

  for (int trip_offset = 0; trip_offset < route.trip_count_; ++trip_offset) {

    if (t_id < active_stop_count) {
      auto const st_idx =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_) + t_id;
      stop_departure = tt.stop_departures_[st_idx];
    }

    // elect leader
    unsigned ballot = __ballot_sync(
        FULL_MASK, (t_id < active_stop_count) && valid(prev_arrival) &&
                       valid(stop_departure) &&
                       (prev_arrival <= stop_departure));
    leader = __ffs(ballot) - 1;

    if (t_id > leader && t_id < active_stop_count) {
      auto const st_idx =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_) + t_id;

      stop_arrival = tt.stop_arrivals_[st_idx];
      bool updated = update_arrival(arrivals, stop_id_t, stop_arrival);
      if (updated) {
        mark(station_marks, stop_id_t);
      }
    }

    if (leader != NO_LEADER) {
      active_stop_count = leader;
    }
    leader = NO_LEADER;
  }
}

__device__ void update_footpaths_dev_scratch(time const* const read_arrivals,
                                             time* const write_arrivals,
                                             unsigned int* station_marks,
                                             device_gpu_timetable const& tt) {

  auto const global_stride = get_global_stride();

  auto foot_idx = get_global_thread_id();
  for (; foot_idx < tt.footpath_count_; foot_idx += global_stride) {
    auto const footpath = tt.footpaths_[foot_idx];

    time const from_arrival = read_arrivals[footpath.from_];
    time const new_arrival = from_arrival + footpath.duration_;

    if (valid(from_arrival) && marked(station_marks, footpath.from_)) {
      bool updated = update_arrival(write_arrivals, footpath.to_, new_arrival);
      if (updated) {
        mark(station_marks, footpath.to_);
      }
    }
  }
}

__device__ void update_routes_dev(time const* const prev_arrivals,
                                  time* const arrivals,
                                  unsigned int* station_marks,
                                  unsigned int* route_marks,
                                  bool* any_station_marked,
                                  device_gpu_timetable const& tt) {

  if (get_global_thread_id() == 0) {
    *any_station_marked = false;
  }

  convert_station_to_route_marks(station_marks, route_marks, any_station_marked,
                                 tt);
  this_grid().sync();

  auto const station_store_size = (tt.stop_count_ / 32) + 1;
  reset_store(station_marks, station_store_size);
  this_grid().sync();

  if (!*any_station_marked) {
    return;
  }

  auto const stride = blockDim.y * gridDim.x;
  auto const start_r_id = threadIdx.y + (blockDim.y * blockIdx.x);
  for (auto r_id = start_r_id; r_id < tt.route_count_; r_id += stride) {
    if (!marked(route_marks, r_id)) {
      continue;
    }

    auto const route = tt.routes_[r_id];
    if (route.stop_count_ <= 32) {
      update_route_smaller32(route, prev_arrivals, arrivals, station_marks, tt);
    } else {
      update_route_larger32(route, prev_arrivals, arrivals, station_marks, tt);
    }
  }

  this_grid().sync();

  auto const store_size = (tt.route_count_ / 32) + 1;
  reset_store(route_marks, store_size);
}

__device__ void init_arrivals_dev(base_query const& query,
                                  device_memory const& device_mem,
                                  device_gpu_timetable const& tt) {
  auto const t_id = get_global_thread_id();

  auto const station_store_size = (tt.stop_count_ / 32) + 1;
  reset_store(device_mem.station_marks_, station_store_size);

  auto const route_store_size = (tt.route_count_ / 32) + 1;
  reset_store(device_mem.route_marks_, route_store_size);

  if (t_id == 0) {
    *device_mem.any_station_marked_ = false;
  }

  if (t_id == 0) {
    device_mem.result_[0][query.source_] = query.source_time_begin_;
    mark(device_mem.station_marks_, query.source_);
  }

  if (t_id < device_mem.additional_start_count_) {
    auto const& add_start = device_mem.additional_starts_[t_id];

    auto const add_start_time = query.source_time_begin_ + add_start.offset_;
    bool updated =
        update_arrival(device_mem.result_[0], add_start.s_id_, add_start_time);

    if (updated) {
      mark(device_mem.station_marks_, add_start.s_id_);
    }
  }
}

__device__ void update_footpaths_dev(device_memory const& device_mem,
                                     raptor_round round_k,
                                     device_gpu_timetable const& tt) {
  time* const arrivals = device_mem.result_[round_k];

  // we must only copy the marked arrivals,
  // since an earlier raptor query might have used a footpath
  // to generate the current arrival, a new optimum from this value
  // would be generated using a double walk -> not correct!
  copy_marked_arrivals(device_mem.footpaths_scratchpad_, arrivals,
                       device_mem.station_marks_, tt);
  this_grid().sync();

  update_footpaths_dev_scratch(device_mem.footpaths_scratchpad_, arrivals,
                               device_mem.station_marks_, tt);
  this_grid().sync();

  if (round_k == max_raptor_round - 1) {
    return;
  }

  time* const next_arrivals = device_mem.result_[round_k + 1];
  copy_and_min_arrivals(next_arrivals, arrivals, tt);
  this_grid().sync();
}

__global__ void gpu_raptor_kernel(base_query const query,
                                  device_memory const device_mem,
                                  device_gpu_timetable const tt) {
  init_arrivals_dev(query, device_mem, tt);
  this_grid().sync();

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    time const* const prev_arrivals = device_mem.result_[round_k - 1];
    time* const arrivals = device_mem.result_[round_k];

    update_routes_dev(prev_arrivals, arrivals, device_mem.station_marks_,
                      device_mem.route_marks_, device_mem.any_station_marked_,
                      tt);
    this_grid().sync();

    update_footpaths_dev(device_mem, round_k, tt);
    this_grid().sync();
  }
}

void invoke_gpu_raptor(d_query const& dq) {
  void* kernel_args[] = {(void*)&dq, (void*)&(dq.mem_->device_),
                         (void*)&(dq.tt_)};
  launch_kernel(gpu_raptor_kernel, kernel_args, dq.mem_->context_,
                dq.mem_->context_.proc_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.proc_stream_);
  cuda_check();

  fetch_arrivals_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();
  cuda_sync_stream(dq.mem_->context_.transfer_stream_);
  cuda_check();
}

}  // namespace motis::raptor