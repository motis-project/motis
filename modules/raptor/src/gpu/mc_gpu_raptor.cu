#include "motis/raptor/gpu/mc_gpu_raptor.cuh"

#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/gpu_mark_store.cuh"
#include "motis/raptor/gpu/raptor_utils.cuh"

#include "motis/raptor/criteria/configs.h"

#include "cooperative_groups.h"

namespace motis::raptor {

using namespace cooperative_groups;

template <typename CriteriaConfig>
__device__ void mc_copy_marked_arrivals(time* const to, time const* const from,
                                        unsigned int* station_marks,
                                        device_gpu_timetable const& tt) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  auto trait_size = CriteriaConfig::trait_size();
  auto max_arrival_idx = tt.stop_count_ * trait_size;
  for (; arr_idx < max_arrival_idx; arr_idx += global_stride) {
    auto const stop_id = arr_idx / trait_size;

    // only copy the values for station + trait offset which are valid
    if (marked(station_marks, stop_id) && valid(from[arr_idx])) {
      to[arr_idx] = from[arr_idx];
    } else {
      to[arr_idx] = invalid<time>;
    }
  }
}

template <typename CriteriaConfig>
__device__ void mc_copy_and_min_arrivals(time* const to, time* const from,
                                         device_gpu_timetable const& tt) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  auto const max_arr_idx = tt.stop_count_ * CriteriaConfig::trait_size();
  for (; arr_idx < max_arr_idx; arr_idx += global_stride) {
    to[arr_idx] = min(from[arr_idx], to[arr_idx]);
  }
}

template<typename CriteriaConfig>
__device__ void mc_update_route_larger32(gpu_route const route,
                                          unsigned int const t_offset,
                                          time const* const prev_arrivals,
                                          time* const arrivals,
                                          unsigned int* station_marks,
                                          device_gpu_timetable const& tt) {
  //TODO implement
}

template<typename CriteriaConfig>
__device__ void mc_update_route_smaller32(gpu_route const route,
                                          unsigned int const t_offset,
                                          time const* const prev_arrivals,
                                          time* const arrivals,
                                          unsigned int* station_marks,
                                          device_gpu_timetable const& tt) {
  //TODO implement
}

template <typename CriteriaConfig>
__device__ void mc_update_footpaths_dev_scratch(
    time const* const read_arrivals, time* const write_arrivals,
    unsigned int* station_marks, device_gpu_timetable const& tt) {

  auto const global_stride = get_global_stride();

  auto arrival_idx = get_global_thread_id();
  auto const trait_size = CriteriaConfig::trait_size();
  auto const max_arr_idx = tt.footpath_count_ * trait_size;

  for (; arrival_idx < max_arr_idx; arrival_idx += global_stride) {
    auto const foot_idx = arrival_idx / trait_size;
    auto const t_offset = arrival_idx % trait_size;

    auto const footpath = tt.footpaths_[foot_idx];

    auto const from_arrival_idx =
        CriteriaConfig::get_arrival_idx(footpath.from_, t_offset);

    time const from_arrival = read_arrivals[from_arrival_idx];
    time const new_arrival = from_arrival + footpath.duration_;

    if (valid(from_arrival) && marked(station_marks, footpath.from_)) {
      auto const to_arrival_idx =
          CriteriaConfig::get_arrival_idx(footpath.to_, t_offset);
      bool updated =
          update_arrival(write_arrivals, to_arrival_idx, new_arrival);
      if (updated) {
        mark(station_marks, footpath.to_);
      }
    }
  }
}

template <typename CriteriaConfig>
__device__ void mc_update_routes_dev(time const* const prev_arrivals,
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

  auto const stride =
      blockDim.y * gridDim.x;  // blockDim.x = 32; blockDim.y = 32; gridDim.x =
                               // 6; => Stride = 32*6 => 192
  auto const start_idx =
      threadIdx.y +
      (blockDim.y *
       blockIdx
           .x);  // threadIdx.y = 1..32 + (blockDim.y = 32 * blockIdx.x = 1..6)
  auto const trait_size = CriteriaConfig::trait_size();
  auto const max_idx = tt.route_count_ * trait_size;
  for (auto idx = start_idx; idx < max_idx; idx += stride) {
    auto const r_id = idx / trait_size;
    if (!marked(route_marks, r_id)) {
      continue;
    }

    auto const route = tt.routes_[r_id];
    auto const t_offset = idx % trait_size;
    if (route.stop_count_ <= 32) {
      mc_update_route_smaller32<CriteriaConfig>(route, t_offset, arrivals,
                                                station_marks, tt);
    } else {
      mc_update_route_larger32<CriteriaConfig>(route, t_offset, arrivals,
                                               station_marks, tt);
    }
  }

  this_grid().sync();
}

template <typename CriteriaConfig>
__device__ void mc_update_footpaths_dev(device_memory const& device_mem,
                                        raptor_round round_k,
                                        device_gpu_timetable const& tt) {
  time* const arrivals = device_mem.result_[round_k];

  // we must only copy the marked arrivals,
  // since an earlier raptor query might have used a footpath
  // to generate the current arrival, a new optimum from this value
  // would be generated using a double walk -> not correct!
  mc_copy_marked_arrivals<CriteriaConfig>(device_mem.footpaths_scratchpad_,
                                          arrivals, device_mem.station_marks_,
                                          tt);
  this_grid().sync();

  mc_update_footpaths_dev_scratch<CriteriaConfig>(
      device_mem.footpaths_scratchpad_, arrivals, device_mem.station_marks_,
      tt);
  this_grid().sync();

  if (round_k == max_raptor_round - 1) {
    return;
  }

  time* const next_arrivals = device_mem.result_[round_k + 1];
  mc_copy_and_min_arrivals<CriteriaConfig>(next_arrivals, arrivals, tt);
  this_grid().sync();
}

template <typename CriteriaConfig>
__device__ void mc_init_arrivals_dev(base_query const& query,
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

  auto const trait_size = CriteriaConfig::trait_size();
  if (t_id < trait_size) {
    auto const arr_idx = CriteriaConfig::get_arrival_idx(query.source_, t_id);
    device_mem.result_[0][arr_idx] = query.source_time_begin_;
    mark(device_mem.station_marks_, query.source_);
  }

  auto req_update_count = device_mem.additional_start_count_ * trait_size;
  auto global_stride = get_global_stride();
  for (auto idx = t_id; idx < req_update_count; idx += global_stride) {
    auto const add_start_idx = idx / trait_size;
    auto const add_start_t_off = idx % trait_size;

    auto const& add_start = device_mem.additional_starts_[add_start_idx];

    auto const add_start_time = query.source_time_begin_ + add_start.offset_;
    auto const add_start_arr_idx =
        CriteriaConfig::get_arrival_idx(add_start.s_id_, add_start_t_off);
    bool updated = update_arrival(device_mem.result_[0], add_start_arr_idx,
                                  add_start_time);

    if (updated) {
      mark(device_mem.station_marks_, add_start.s_id_);
    }
  }
}

template <typename CriteriaConfig>
__global__ void mc_gpu_raptor_kernel(base_query const query,
                                     device_memory const device_mem,
                                     device_gpu_timetable const tt) {
  mc_init_arrivals_dev<CriteriaConfig>(query, device_mem, tt);
  this_grid().sync();

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    time const* const prev_arrivals = device_mem.result_[round_k - 1];
    time* const arrivals = device_mem.result_[round_k];

    mc_update_routes_dev<CriteriaConfig>(
        prev_arrivals, arrivals, device_mem.station_marks_,
        device_mem.route_marks_, device_mem.any_station_marked_, tt);

    this_grid().sync();

    mc_update_footpaths_dev<CriteriaConfig>(device_mem, round_k, tt);

    this_grid().sync();
  }
}

template <typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const& dq) {
  void* kernel_args[] = {(void*)&dq, (void*)(dq.mem_->active_device_),
                         (void*)&dq.tt_};
  launch_kernel(mc_gpu_raptor_kernel<CriteriaConfig>, kernel_args,
                dq.mem_->context_, dq.mem_->context_.proc_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.proc_stream_);
  cuda_check();

  fetch_arrivals_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.transfer_stream_);
  cuda_check();
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_GPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor