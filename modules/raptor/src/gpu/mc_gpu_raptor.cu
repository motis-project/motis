#include "motis/raptor/gpu/mc_gpu_raptor.cuh"

#include <iostream>

#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/gpu_mark_store.cuh"
#include "motis/raptor/gpu/raptor_utils.cuh"
#include "motis/raptor/gpu/update_arrivals.cuh"

#include "motis/raptor/criteria/configs.h"

#include "cooperative_groups.h"

namespace motis::raptor {

using namespace cooperative_groups;

// leader type must be unsigned 32bit
// no leader is a zero ballot vote (all 0) minus 1 => with underflow all 1's
constexpr unsigned int FULL_MASK = 0xFFFFffff;
constexpr unsigned int NO_LEADER = FULL_MASK;

template <typename CriteriaConfig>
__device__ occ_t get_moc(typename CriteriaConfig::CriteriaData const& d) {
  return 3;
}

template <>
__device__ occ_t get_moc<MaxOccupancy>(MaxOccupancy ::CriteriaData const& d) {
  return d.max_occupancy_;
}

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

__device__ __forceinline__ unsigned get_criteria_propagation_mask(
    unsigned const leader, unsigned const stop_count) {
  auto const stops_to_update = stop_count - leader - 1;
  auto const mask = (1 << stops_to_update) - 1;
  return (mask << (leader + 1));
}

template <typename CriteriaConfig>
__device__ void mc_update_route_larger32(
    route_id const r_id, gpu_route const route, unsigned int const t_offset,
    time const* const prev_arrivals, time* const arrivals,
    unsigned int* station_marks, device_gpu_timetable const& tt) {

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

template <typename CriteriaConfig>
__device__ void mc_update_route_smaller32(
    route_id const r_id, gpu_route const route, unsigned int const t_offset,
    time const* const prev_arrivals, time* const arrivals,
    unsigned int* station_marks, device_gpu_timetable const& tt) {

  auto const t_id = threadIdx.x;

  stop_id s_id = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  arrival_id stop_arr_idx = invalid<arrival_id>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;
  typename CriteriaConfig::CriteriaData aggregate{};

  unsigned leader = route.stop_count_;
  unsigned int active_stop_count = route.stop_count_;

  if (t_id < active_stop_count) {
    s_id = tt.route_stops_[route.index_to_route_stops_ + t_id];
    stop_arr_idx = CriteriaConfig::get_arrival_idx(s_id, t_offset);
    prev_arrival = prev_arrivals[stop_arr_idx];
  }

  // we skip updates if there is no feasible departure station
  //  on this route with the given trait offset
  if (!__any_sync(FULL_MASK, valid(prev_arrivals))) {
    return;
  }

  for (trip_id trip_offset = 0; trip_offset < route.trip_count_;
       ++trip_offset) {
    if (t_id < active_stop_count) {
      auto const st_index =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_) + t_id;
      stop_departure = tt.stop_departures_[st_index];
    }

    unsigned ballot = __ballot_sync(
        FULL_MASK, (t_id < active_stop_count) && valid(prev_arrival) &&
                       valid(stop_departure) &&
                       (prev_arrival <= stop_departure));
    if (t_id == 0)
      printf(
          "Ballot Mask for r_id: %i, t_offset: %i, trip_offset: "
          "%i;\t%x\n",
          r_id, t_offset, trip_offset, ballot);

    leader =
        __ffs(ballot) - 1;  // index of the first departure location on route

    unsigned criteria_mask =
        get_criteria_propagation_mask(leader, active_stop_count);

    if (t_id == 0) {
      printf(
          "Criteria Mask for r_id: %i, t_offset: %i, trip_offset: "
          "%i;\t%x\n",
          r_id, t_offset, trip_offset, criteria_mask);
    }

    if (t_id > leader && t_id < active_stop_count) {
      auto const st_index =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_) + t_id;

      stop_arrival = tt.stop_arrivals_[st_index];
      auto const is_departure_stop = (((1 << t_id) & ballot) >> t_id);
      if (is_departure_stop)
        printf(
            "Is Departure Stop: r_id: %i\tt_offset: %i;\t trip_id: %i\tt_id: "
            "%i\ts_id: %i\n",
            r_id, t_offset, trip_offset, t_id, s_id);

      if (!is_departure_stop) {
        CriteriaConfig::update_traits_aggregate(aggregate, tt, r_id,
                                                trip_offset, t_id, st_index);
      }

      // propagate the additional criteria attributes
      for (int idx = leader + 1; idx < active_stop_count; ++idx) {
        // internally uses __shfl_up_sync to propagate the criteria values
        //  along the traits while allowing for max/min/sum operations
        CriteriaConfig::propagate_and_merge_if_needed(
            criteria_mask, aggregate, !is_departure_stop && idx <= t_id);
      }

      printf(
          "\nt_id: %i\tr_id: %i\tt_offset: %i\ttrip_id: %i\tfound moc "
          "for "
          "s_id: %i\tmoc: %i\n",
          t_id, r_id, t_offset, trip_offset, s_id,
          get_moc<CriteriaConfig>(aggregate));

      if (CriteriaConfig::is_update_required(aggregate, t_offset)) {
        bool updated = update_arrival(arrivals, stop_arr_idx, stop_arrival);
        if (updated) {
          printf(
              "\nt_id: %i\tr_id: %i\tt_offset: %i\ttrip_id: %i\twrite update "
              "for "
              "s_id: %i\tto arr idx: %i\tarr_time: %i\n",
              t_id, r_id, t_offset, trip_offset, s_id, stop_arr_idx,
              stop_arrival);
          mark(station_marks, s_id);
        }
      }
    }

    CriteriaConfig::reset_traits_aggregate(aggregate);
  }
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
      mc_update_route_smaller32<CriteriaConfig>(
          r_id, route, t_offset, prev_arrivals, arrivals, station_marks, tt);
    } else {
      // mc_update_route_larger32<CriteriaConfig>(
      //     r_id, route, t_offset, prev_arrivals, arrivals, station_marks, tt);
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

  for (raptor_round round_k = 1; round_k < 2; ++round_k) {
    time const* const prev_arrivals = device_mem.result_[round_k - 1];
    time* const arrivals = device_mem.result_[round_k];

    mc_update_routes_dev<CriteriaConfig>(
        prev_arrivals, arrivals, device_mem.station_marks_,
        device_mem.route_marks_, device_mem.any_station_marked_, tt);

    this_grid().sync();

    //  mc_update_footpaths_dev<CriteriaConfig>(device_mem, round_k, tt);

    //  this_grid().sync();
  }
}

template <typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const& dq) {
  printf("arrivals count: %i", dq.result().arrival_times_count_);

  void* kernel_args[] = {(void*)&dq, (void*)(dq.mem_->active_device_),
                         (void*)&dq.tt_};

  launch_kernel(mc_gpu_raptor_kernel<CriteriaConfig>, kernel_args,
                dq.mem_->context_, dq.mem_->context_.proc_stream_,
                dq.criteria_config_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.proc_stream_);
  cuda_check();

  fetch_arrivals_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.transfer_stream_);
  cuda_check();
}

#define GENERATE_LAUNCH_CONFIG_FUNCTION(VAL, ACCESSOR)             \
  template <>                                                      \
  std::tuple<int, int> get_mc_gpu_launch_config<VAL>() {           \
    int block_size, grid_size;                                     \
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,    \
                                       mc_gpu_raptor_kernel<VAL>); \
    return std::make_tuple(grid_size, block_size);                 \
  }

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(GENERATE_LAUNCH_CONFIG_FUNCTION,
                                   raptor_criteria_config)

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_GPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor