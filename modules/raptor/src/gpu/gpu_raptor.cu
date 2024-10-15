#include "motis/raptor/gpu/gpu_raptor.cuh"

#include "motis/raptor/gpu/gpu_mark_store.cuh"
#include "motis/raptor/gpu/raptor_utils.cuh"
#include "motis/raptor/gpu/update_arrivals.cuh"

#include "cooperative_groups.h"
#include "cuda_profiler_api.h"

namespace motis::raptor {

using namespace cooperative_groups;

// leader type must be unsigned 32bit
// no leader is a zero ballot vote (all 0) minus 1 => with underflow all 1's
constexpr unsigned int FULL_MASK = 0xFFFFffff;
constexpr unsigned int NO_LEADER = FULL_MASK;

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
                                      uint32_t* station_marks,
                                      device_gpu_timetable const& tt) {
  auto const t_id = threadIdx.x;

  stop_id stop_id_t = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;
  time transfer_time = invalid<time>;

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
        transfer_time = tt.transfer_times_[stop_id_t];
      }

      // get the current stage leader
      unsigned int ballot = __ballot_sync(
          FULL_MASK, (stage_id < active_stop_count) && valid(prev_arrival) &&
                         valid(stop_departure) &&
                         (prev_arrival + transfer_time <= stop_departure));
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
                                       uint32_t* station_marks,
                                       device_gpu_timetable const& tt) {
  auto const t_id = threadIdx.x;

  stop_id stop_id_t = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;
  time transfer_time = invalid<time>;

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
      transfer_time = tt.transfer_times_[stop_id_t];
    }

    // elect leader
    unsigned ballot = __ballot_sync(
        FULL_MASK, (t_id < active_stop_count) && valid(prev_arrival) &&
                       valid(stop_departure) &&
                       (prev_arrival + transfer_time <= stop_departure));
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
                                             uint32_t* station_marks,
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
                                  time* const arrivals, uint32_t* station_marks,
                                  uint32_t* route_marks,
                                  device_gpu_timetable const& tt) {

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

  auto const start_time =
      query.source_time_begin_ - tt.transfer_times_[query.source_];

  if (t_id == 0) {
    device_mem.result_[0][query.source_] = start_time;
    mark(device_mem.station_marks_, query.source_);
  }

  if (t_id < device_mem.additional_start_count_) {
    auto const& add_start = device_mem.additional_starts_[t_id];

    auto const add_start_time = start_time + add_start.offset_;
    bool updated =
        update_arrival(device_mem.result_[0], add_start.s_id_, add_start_time);

    if (updated) {
      mark(device_mem.station_marks_, add_start.s_id_);
    }
  }
}

__device__ void update_footpaths_dev(device_memory const& device_mem,
                                     raptor_round const round_k,
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
    if (get_global_thread_id() == 0) {
      *(device_mem.any_station_marked_) = false;
    }
    this_grid().sync();

    convert_station_to_route_marks(device_mem.station_marks_,
                                   device_mem.route_marks_,
                                   device_mem.any_station_marked_, tt);
    this_grid().sync();

    auto const station_store_size = (tt.stop_count_ / 32) + 1;
    reset_store(device_mem.station_marks_, station_store_size);
    this_grid().sync();

    if (!(*device_mem.any_station_marked_)) {
      return;
    }

    time const* const prev_arrivals = device_mem.result_[round_k - 1];
    time* const arrivals = device_mem.result_[round_k];

    update_routes_dev(prev_arrivals, arrivals, device_mem.station_marks_,
                      device_mem.route_marks_, tt);
    this_grid().sync();

    update_footpaths_dev(device_mem, round_k, tt);
    this_grid().sync();
  }
}

std::pair<dim3, dim3> get_gpu_raptor_launch_parameters(
    device_id const device_id, int32_t const concurrency_per_device) {
  cudaSetDevice(device_id);
  cuda_check();

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);
  cuda_check();

  utl::verify(
      prop.warpSize == 32,
      "Warp Size must be 32! Otherwise the gRAPTOR algorithm will not work.");

  int min_grid_size = 0;
  int block_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     gpu_raptor_kernel, 0, 0);

  dim3 threads_per_block(prop.warpSize, block_size / prop.warpSize, 1);
  dim3 grid(min_grid_size / concurrency_per_device, 1, 1);

  return {threads_per_block, grid};
}

void invoke_gpu_raptor(d_query const& dq) {
  void* kernel_args[] = {(void*)&dq, (void*)(dq.mem_->active_device_),
                         (void*)&(dq.tt_)};
  launch_kernel(gpu_raptor_kernel, kernel_args, dq.mem_->context_,
                dq.mem_->context_.proc_stream_, dq.criteria_config_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.proc_stream_);
  cuda_check();

  fetch_arrivals_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.transfer_stream_);
  cuda_check();
}

}  // namespace motis::raptor