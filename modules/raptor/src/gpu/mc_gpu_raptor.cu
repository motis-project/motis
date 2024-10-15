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
__device__ void print_aggregate(CriteriaConfig const& d) {}

template <>
__device__ void print_aggregate<MaxTransferClass>(MaxTransferClass const& d) {
  printf("dep_off %i\tprev %i\tdep %i\ttt %i\tfast %i\tslow %i\tmtc %i\n",
         d._dep_offset, d._prev_arr, d._stop_dep, d._regular_tt, d._fast_tt,
         d._slow_tt, d.active_transfer_class_);
}

template <typename CriteriaConfig>
__device__ void mc_copy_marked_arrivals(time* const to, time const* const from,
                                        unsigned int* station_marks,
                                        device_gpu_timetable const& tt) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  auto trait_size = CriteriaConfig::TRAITS_SIZE;
  auto max_arrival_idx = tt.stop_count_ * trait_size;
  for (; arr_idx < max_arrival_idx; arr_idx += global_stride) {
    // auto const stop_id = arr_idx / trait_size;

    // only copy the values for station + trait offset which are valid
    if (marked(station_marks, arr_idx) && valid(from[arr_idx])) {
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
  auto const max_arr_idx = tt.stop_count_ * CriteriaConfig::TRAITS_SIZE;
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

__device__ __forceinline__ unsigned get_last_departure_stop(
    unsigned criteria_mask) {
  unsigned const rev = __brev(criteria_mask);
  unsigned const last = 32 - __ffs(rev);
  return last;
}

__device__ __forceinline__ void reset_ea(time* ea, size_t const size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();
  for (auto idx = t_id; idx < size; idx += stride) {
    ea[idx] = invalid<time>;
  }
}

__device__ __forceinline__ stop_offset get_closest_departure_sti(
    unsigned const ballot, stop_offset const arr_offset,
    uint32_t current_stage = 0,
    stop_offset const last_known_dep_stop_offset = invalid<stop_offset>) {
  assert(arr_offset > 0);
  // the shifted mask only contains the ballot results for all stops which are
  // located before the stop under investigation
  // it is then reversed so that the stop directly before the arrival stop
  // is at position 0 and the stop before that at position 1 and so forth
  // therefore, the index of the first 1 found in the mask represents the
  // distance in (measured in stops) to the departure stop
  // which can then be used to determine the departure offset
  auto const shift_by = 32 - arr_offset;
  auto const shifted = ballot << shift_by;
  auto const rev_shifted = __brev(shifted);
  auto const first_dep_idx = __ffs(rev_shifted);
  if (first_dep_idx > 0) {
    return (current_stage << 5) + arr_offset - first_dep_idx;
  } else {
    // this can only happen for routes with > 32 stops
    //  if there exists a departure stop in a stage before the current
    // then return the last known dep stop offset
    assert(valid(last_known_dep_stop_offset));
    return last_known_dep_stop_offset;
  }
}

template <typename CriteriaConfig>
__device__ __forceinline__ time get_earliest_arrival(
    const time* const earliest_arrivals, stop_id const target_stop_id,
    stop_id const current_stop_id, trait_id const write_to_offset) {

  arrival_id const current_idx =
      CriteriaConfig::get_arrival_idx(current_stop_id, write_to_offset);
  time const stop_ea = earliest_arrivals[current_idx];
  auto const target_arr_idx =
      CriteriaConfig::get_arrival_idx(target_stop_id, write_to_offset);
  time const target_ea = earliest_arrivals[target_arr_idx];
  return umin(stop_ea, target_ea);
}

template <typename CriteriaConfig>
__device__ void mc_update_route_larger32(
    route_id const r_id, gpu_route const route, trait_id const t_offset,
    time const* const prev_arrivals, time* const arrivals,
    time* const earliest_arrivals, stop_id const target_stop_id,
    uint32_t* station_marks, device_gpu_timetable const& tt,
    uint32_t* fp_marks, uint32_t const fp_offset) {

  auto const t_id = threadIdx.x;

  stop_id stop_id_t = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;

  CriteriaConfig aggregate{&route, t_offset};
  unsigned last_known_dep_stop = invalid<unsigned>;

  int active_stop_count = route.stop_count_;

  // this is ceil(stop_count / 32)
  int const stage_count = (route.stop_count_ + (32 - 1)) >> 5;
  int active_stage_count = stage_count;

  unsigned int leader = NO_LEADER;
  unsigned int any_arrival = 0;
  unsigned int criteria_mask = 0;

  for (int trip_offset = 0; trip_offset < route.trip_count_; ++trip_offset) {
    if (CriteriaConfig::USES_SHFL_CALC) aggregate.reset(trip_offset, t_offset);

    for (uint32_t current_stage = 0; current_stage < active_stage_count;
         ++current_stage) {
      if (!CriteriaConfig::USES_SHFL_CALC)
        aggregate.reset(trip_offset, t_offset);

      uint32_t stage_id = (current_stage << 5) + t_id;  // stage_id ^= stop_id

      // load the prev arrivals for the current stage
      if (stage_id < active_stop_count) {
        stop_id_t = tt.route_stops_[route.index_to_route_stops_ + stage_id];
        auto const stop_arr_idx =
            CriteriaConfig::get_arrival_idx(stop_id_t, t_offset);
        prev_arrival = prev_arrivals[stop_arr_idx];
      }

      any_arrival |= __any_sync(FULL_MASK, valid(prev_arrival));

      if (current_stage == active_stage_count - 1 && !any_arrival) {
        return;
      }

      if (!any_arrival) {
        continue;
      }

      // load the stop times for the current stage
      bool departure_feasible = false;
      if (stage_id < active_stop_count) {
        auto const st_idx = route.index_to_stop_times_ +
                            (trip_offset * route.stop_count_) + stage_id;
        stop_departure = tt.stop_departures_[st_idx];
        departure_feasible = aggregate.check_and_set_departure_stop(
            tt, stage_id, stop_id_t, prev_arrival, stop_departure);
      }

      // get the current stage leader
      unsigned int ballot = __ballot_sync(
          FULL_MASK, (stage_id < active_stop_count) && valid(prev_arrival) &&
                         valid(stop_departure) && departure_feasible);

      // index of first possible departure station on this stage
      leader = __ffs(ballot) - 1;

      auto const stop_count_on_stage =
          active_stop_count < ((current_stage + 1) << 5)
              ? (active_stop_count - (current_stage << 5))
              : 32;

      if (valid(last_known_dep_stop)) {
        // no leader in current stags; though maybe there is a leader in the
        //   previous stage
        //        printf("Active Stop count %i;\ton stage: %i\n",
        //        stop_count_on_stage, current_stage);
        criteria_mask = (1 << stop_count_on_stage) - 1;
      } else if (leader != NO_LEADER) {
        criteria_mask =
            get_criteria_propagation_mask(leader, stop_count_on_stage);
      } else {
        // no departure stop up to now
        continue;
      }

      if (leader > 0 && valid(last_known_dep_stop) &&
          CriteriaConfig::USES_SHFL_CALC) {
        // in a case where there is no departure location in this stage
        // or the departure location (leader) is after this stop
        // and there is a valid known departure stop from teh previous stage
        // carry over the aggregate from the last stop in the previous stage
        // to the first stop in this stage
        unsigned carry_mask = (1 << 31) | 1;
        aggregate.carry_to_next_stage(carry_mask);
      }

      auto const has_carry_value = t_id == 0 && valid(last_known_dep_stop);
      if (!has_carry_value) {

        // don't reset if this has a carry value even if it is a departure
        // station
        //  as arrival time might be improved through the carry value
        aggregate.reset(trip_offset, t_offset);
      }

      if (leader != NO_LEADER) {
        // adjust the determined departure location to the current stage
        leader += current_stage << 5;
      }

      // update this stage if there is a leader or a known dep stop from
      // one of the previous stages
      bool wrote_satis_time = false;
      if ((leader != NO_LEADER || valid(last_known_dep_stop)) &&
          stage_id < active_stop_count) {

        if ((leader != NO_LEADER && stage_id > leader) ||
            (valid(last_known_dep_stop) && stage_id > last_known_dep_stop)) {
          auto const first_sti =
              route.index_to_stop_times_ + (trip_offset * route.stop_count_);
          auto const st_idx = first_sti + stage_id;
          stop_arrival = tt.stop_arrivals_[st_idx];

          if (CriteriaConfig::USES_SHFL_CALC) {
            // is_departure_stop is local to the current stage
            auto const is_departure_stop = (((1 << t_id) & ballot) >> t_id);

            aggregate.update_from_stop(tt, t_id, st_idx);

            // propagate the additional criteria attributes
            for (uint32_t idx = __ffs(criteria_mask); idx < stop_count_on_stage;
                 ++idx) {
              // internally uses __shfl_up_sync to propagate the criteria values
              //  along the traits while allowing for max/min/sum operations
              aggregate.propagate_along_warp(
                  criteria_mask, is_departure_stop,
                  idx <= t_id
                      // prevent write update if this has carry value
                      && !has_carry_value);
            }
          } else {

            auto const dep_offset = get_closest_departure_sti(
                ballot, t_id, current_stage, last_known_dep_stop);
            auto const dep_sti = first_sti + dep_offset;
            aggregate.calculate(tt, prev_arrivals, dep_sti, st_idx);
          }

          auto const write_to_offset = aggregate.get_write_to_trait_id();

          if (valid(write_to_offset)) {
            auto const earliest_arrival = get_earliest_arrival<CriteriaConfig>(
                earliest_arrivals, target_stop_id, stop_id_t, write_to_offset);

            if (stop_arrival < earliest_arrival) {
              auto const write_to_arr_idx =
                  CriteriaConfig::get_arrival_idx(stop_id_t, write_to_offset);
              bool updated =
                  update_arrival(arrivals, write_to_arr_idx, stop_arrival);
              if (updated) {
                wrote_satis_time = aggregate.is_satisfied(t_offset);

                update_arrival(earliest_arrivals, write_to_arr_idx,
                               stop_arrival);
                mark(station_marks, write_to_arr_idx);
                unmark(fp_marks, fp_offset + write_to_arr_idx);
              }
            }
          }
        }
      }

      if (leader != NO_LEADER) {
        if (current_stage == active_stage_count - 1) {
          // at the last stage check the stop satisfaction and reduce asc if
          // possible
          time satisfied_ea = invalid<time>;
          if ((1 << t_id) & criteria_mask) {
            auto satisfied_arr_idx =
                CriteriaConfig::get_arrival_idx(stop_id_t, t_offset);
            satisfied_ea = get_earliest_arrival<CriteriaConfig>(
                earliest_arrivals, stop_id_t, target_stop_id, t_offset);
          }

          auto satisfied_ballot = __ballot_sync(
              FULL_MASK,
              stage_id < active_stop_count &&
                  (wrote_satis_time ||
                   (valid(stop_arrival) && satisfied_ea <= stop_arrival)));

          auto stage_asc = active_stop_count - (current_stage << 5);
          auto const helper_mask =
              stage_asc < 32 ? __brev((1 << (32 - stage_asc)) - 1) : 0;
          auto const inverted_ballot = ~(satisfied_ballot | helper_mask);
          auto leading_zero_count = __clz(inverted_ballot);
          //          auto const init_lzc = leading_zero_count;
          leading_zero_count -= (32 - stage_asc);
          active_stop_count -= leading_zero_count;

          //          auto const init_stage_asc = stage_asc;
          stage_asc -= leading_zero_count;

          if (stage_asc == 0) {
            active_stage_count -= 1;
          }

        } else {
          // there is a leader in the current stage; therefore safe the last
          // possible departure stop for updates to the next stage
          last_known_dep_stop = get_closest_departure_sti(
              ballot, 32, current_stage, last_known_dep_stop);
          //          last_known_dep_stop += current_stage << 5;
        }
      }
    }
  }
}

template <typename CriteriaConfig>
__device__ void mc_update_route_smaller32(
    route_id const r_id, gpu_route const route, trait_id const t_offset,
    time const* const prev_arrivals, time* const arrivals,
    time* const earliest_arrivals, stop_id const target_stop_id,
    uint32_t* station_marks, device_gpu_timetable const& tt,
    uint32_t* fp_marks, uint32_t const fp_offset) {

  auto const t_id = threadIdx.x;

  stop_id s_id = invalid<stop_id>;
  time prev_arrival = invalid<time>;
  time stop_arrival = invalid<time>;
  time stop_departure = invalid<time>;

  CriteriaConfig aggregate{&route, t_offset};

  unsigned leader = route.stop_count_;
  unsigned int active_stop_count = route.stop_count_;

  if (t_id < active_stop_count) {
    s_id = tt.route_stops_[route.index_to_route_stops_ + t_id];
    auto const stop_arr_idx = CriteriaConfig::get_arrival_idx(s_id, t_offset);
    prev_arrival = prev_arrivals[stop_arr_idx];
  }

  // we skip updates if there is no feasible departure station
  //  on this route with the given trait offset
  if (!__any_sync(FULL_MASK, valid(prev_arrivals))) {
    return;
  }

  for (trip_id trip_offset = 0; trip_offset < route.trip_count_;
       ++trip_offset) {

    aggregate.reset(trip_offset, t_offset);

    bool departure_feasible = false;
    if (t_id < active_stop_count) {
      auto const st_index =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_) + t_id;
      stop_departure = tt.stop_departures_[st_index];
      departure_feasible = aggregate.check_and_set_departure_stop(
          tt, t_id, s_id, prev_arrival, stop_departure);
    }

    unsigned ballot = __ballot_sync(
        FULL_MASK, (t_id < active_stop_count) && valid(prev_arrival) &&
                       valid(stop_departure) && departure_feasible);

    // index of the first departure location on route
    leader = __ffs(ballot) - 1;

    if (leader == NO_LEADER) continue;  // No feasible departure on this trip

    unsigned criteria_mask =
        get_criteria_propagation_mask(leader, active_stop_count);

    bool wrote_satis_time = false;
    if (t_id > leader && t_id < active_stop_count) {
      auto const first_sti =
          route.index_to_stop_times_ + (trip_offset * route.stop_count_);
      auto const st_index = first_sti + t_id;

      stop_arrival = tt.stop_arrivals_[st_index];

      if (CriteriaConfig::USES_SHFL_CALC) {
        auto const is_departure_stop = (((1 << t_id) & ballot) >> t_id);
        aggregate.update_from_stop(tt, t_id /* == stop offset */, st_index);

        // propagate the additional criteria attributes
        for (uint32_t idx = leader + 1; idx < active_stop_count; ++idx) {
          // internally uses __shfl_up_sync to propagate the criteria values
          //  along the traits while allowing for max/min/sum operations
          aggregate.propagate_along_warp(criteria_mask, is_departure_stop,
                                         idx <= t_id);
        }
      } else {
        auto const dep_offset = get_closest_departure_sti(ballot, t_id);
        auto const dep_sti = first_sti + dep_offset;
        aggregate.calculate(tt, prev_arrivals, dep_sti, st_index);
      }

      auto const write_to_offset = aggregate.get_write_to_trait_id();
      if (valid(write_to_offset) && t_id > leader) {

        // Note: Earliest Arrival may, when reaching this point not be the
        //       'earliest arrival' at this stop, but it gives a sufficient
        //       upper bound and allows preventing arrival time which are the
        //       same as for one round earlier
        auto const earliest_arrival = get_earliest_arrival<CriteriaConfig>(
            earliest_arrivals, target_stop_id, s_id, write_to_offset);

        if (stop_arrival < earliest_arrival) {
          auto const write_to_idx =
              CriteriaConfig::get_arrival_idx(s_id, write_to_offset);
          bool updated = update_arrival(arrivals, write_to_idx, stop_arrival);
          if (updated) {
            wrote_satis_time = aggregate.is_satisfied(t_offset);

            update_arrival(earliest_arrivals, write_to_idx, stop_arrival);
            mark(station_marks, write_to_idx);
            unmark(fp_marks, fp_offset + write_to_idx);
          }
        }
      }
    }

    // check if stops on route are satisfied
    if (leader != NO_LEADER) {
      time satisfied_ea = invalid<time>;
      if ((1 << t_id) & criteria_mask) {
        auto satisfied_arr_idx =
            CriteriaConfig::get_arrival_idx(s_id, t_offset);
        satisfied_ea = get_earliest_arrival<CriteriaConfig>(
            earliest_arrivals, s_id, target_stop_id, t_offset);
      }

      auto const satisfied_ballot = __ballot_sync(
          FULL_MASK, t_id < active_stop_count &&
                         (wrote_satis_time || (valid(stop_arrival) &&
                                               satisfied_ea <= stop_arrival)));

      // not satisfied yet but there's a chance we can reduce
      // the number of stops to be scanned on the next trip
      auto const helper_mask = __brev((1 << (32 - active_stop_count)) - 1);
      auto inverted_ballot = ~(satisfied_ballot | helper_mask);
      auto leading_zero_count = __clz(inverted_ballot);
      //      auto const initial_clz = leading_zero_count;
      leading_zero_count -= (32 - active_stop_count);

      //      auto const initial_acs = active_stop_count;
      active_stop_count -= leading_zero_count;

      // if every stop is satisfied we can skip further updates
      if ((1 << route.stop_count_) - 1 == satisfied_ballot) {
        break;
      }
    }
    leader = NO_LEADER;
  }
}

template <typename CriteriaConfig>
__device__ void mc_update_footpaths_dev_scratch(
    time const* const read_arrivals, time* const write_arrivals,
    time* const earliest_arrivals, stop_id const target_stop_id,
    uint32_t* station_marks, device_gpu_timetable const& tt,
    uint32_t* fp_marks, uint32_t const fp_offset) {

  auto const global_stride = get_global_stride();

  auto arrival_idx = get_global_thread_id();
  auto const trait_size = CriteriaConfig::TRAITS_SIZE;
  auto const max_arr_idx = tt.footpath_count_ * trait_size;
  auto const target_arr_idx = CriteriaConfig::get_arrival_idx(target_stop_id);

  for (; arrival_idx < max_arr_idx; arrival_idx += global_stride) {
    auto const foot_idx = arrival_idx / trait_size;
    auto const t_offset = arrival_idx % trait_size;

    auto const footpath = tt.footpaths_[foot_idx];

    auto const from_arrival_idx =
        CriteriaConfig::get_arrival_idx(footpath.from_, t_offset);
    auto const to_arrival_idx =
        CriteriaConfig::get_arrival_idx(footpath.to_, t_offset);

    time const from_arrival = read_arrivals[from_arrival_idx];
    if (valid(from_arrival)
        && !marked(fp_marks, fp_offset + from_arrival_idx)) {
      time const new_arrival = from_arrival + footpath.duration_;

      // this give potentially just an upper bound and not the real
      //  earliest arrival value at the time the update is written
      time const to_stop_ea = earliest_arrivals[to_arrival_idx];
      time const target_ea = earliest_arrivals[target_arr_idx + t_offset];
      time const earliest_arrival = umin(to_stop_ea, target_ea);

      if (marked(station_marks, from_arrival_idx) &&
          new_arrival < earliest_arrival) {
        bool updated =
            update_arrival(write_arrivals, to_arrival_idx, new_arrival);
        if (updated) {
          update_arrival(earliest_arrivals, to_arrival_idx, new_arrival);
          mark(station_marks, to_arrival_idx);
          mark(fp_marks, fp_offset + to_arrival_idx);
        }
      }
    }
  }
}

template <typename CriteriaConfig>
__device__ void perform_arrival_sweeping(stop_id const stop_count,
                                         time* const arrivals, time* const ea,
                                         uint32_t* station_marks) {
  auto const global_stride = get_global_stride();

  auto s_id = get_global_thread_id();
  // one thread scans all arrivals on one stop
  for (; s_id < stop_count; s_id += global_stride) {
    CriteriaConfig::perform_stop_arrival_sweeping_gpu(s_id, arrivals, station_marks);
  }
}

template <typename CriteriaConfig>
__device__ void mc_update_routes_dev(device_memory const& device_mem,
                                     raptor_round const round_k,
                                     stop_id const target_stop_id,
                                     device_gpu_timetable const& tt) {

  time const* const prev_arrivals = device_mem.result_[round_k - 1];
  time* const arrivals = device_mem.result_[round_k];
  uint32_t* route_marks = device_mem.route_marks_;
  uint32_t* station_marks = device_mem.station_marks_;
  time* const earliest_arrivals = device_mem.earliest_arrivals_;

  // blockDim.x = 32; blockDim.y = 32; gridDim.x =
  // 6; => Stride = 32*6 => 192
  auto const stride = blockDim.y * gridDim.x;
  // threadIdx.y = 1..32 + (blockDim.y = 32 * blockIdx.x = 1..6)
  auto const start_idx = threadIdx.y + (blockDim.y * blockIdx.x);

  auto const trait_size = CriteriaConfig::TRAITS_SIZE;

  auto const fp_offset = (round_k - 1) * trait_size * tt.stop_count_;

  auto const max_idx = tt.route_count_ * trait_size;
  for (auto idx = start_idx; idx < max_idx; idx += stride) {
    if (!marked(route_marks, idx)) {
      continue;
    }

    auto const r_id = idx / trait_size;
    auto const route = tt.routes_[r_id];
    auto const t_offset = idx % trait_size;

    if (route.stop_count_ <= 32) {
      mc_update_route_smaller32<CriteriaConfig>(
          r_id, route, t_offset, prev_arrivals, arrivals, earliest_arrivals,
          target_stop_id, station_marks, tt,
          device_mem.fp_marks_, fp_offset);
    } else {
      mc_update_route_larger32<CriteriaConfig>(
          r_id, route, t_offset, prev_arrivals, arrivals, earliest_arrivals,
          target_stop_id, station_marks, tt,
          device_mem.fp_marks_, fp_offset);
    }
  }

  this_grid().sync();

  auto const store_size = (max_idx / 32) + 1;
  reset_store(route_marks, store_size);
}

template <typename CriteriaConfig>
__device__ void mc_update_footpaths_dev(device_memory const& device_mem,
                                        raptor_round const round_k,
                                        stop_id const target_stop_id,
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

  auto const fp_offset = (round_k - 1)
                         * CriteriaConfig::TRAITS_SIZE
                         * tt.stop_count_;

  mc_update_footpaths_dev_scratch<CriteriaConfig>(
      device_mem.footpaths_scratchpad_, arrivals, device_mem.earliest_arrivals_,
      target_stop_id, device_mem.station_marks_, tt,
      device_mem.fp_marks_, fp_offset);
  this_grid().sync();
}

template <typename CriteriaConfig>
__device__ void mc_init_arrivals_dev(base_query const& query,
                                     device_memory const& device_mem,
                                     device_gpu_timetable const& tt) {
  auto const t_id = get_global_thread_id();
  auto const trait_size = CriteriaConfig::TRAITS_SIZE;
  auto const sweep_block_size = CriteriaConfig::SWEEP_BLOCK_SIZE;

  auto write_to_trait_blocks = [&](time* const arrivals, stop_id const s_id,
                                   time const source_time) {
    auto const first_arr_idx = s_id * trait_size;
    auto const last_arr_idx = first_arr_idx + trait_size;

    for (auto arr_idx = first_arr_idx; arr_idx < last_arr_idx;
         arr_idx += sweep_block_size) {
      auto const t_offset = arr_idx - first_arr_idx;
      auto const trans_time =
          CriteriaConfig::get_transfer_time(tt, t_offset, s_id);
      auto const arriv_time = source_time - trans_time;

      bool updated = update_arrival(arrivals, arr_idx, arriv_time);
      if (updated) {
        mark(device_mem.station_marks_, arr_idx);
      }
    }
  };

  if (t_id == 0) {
    write_to_trait_blocks(device_mem.result_[0], query.source_,
                          query.source_time_begin_);
  }

  auto req_update_count = device_mem.additional_start_count_;
  auto global_stride = get_global_stride();
  for (auto add_start_idx = t_id; add_start_idx < req_update_count;
       add_start_idx += global_stride) {

    auto const& add_start = device_mem.additional_starts_[add_start_idx];

    auto const add_start_time = query.source_time_begin_ + add_start.offset_;

    write_to_trait_blocks(device_mem.result_[0], add_start.s_id_,
                          add_start_time);
  }
}

template <typename CriteriaConfig>
__global__ void mc_gpu_raptor_kernel(base_query const query,
                                     device_memory const device_mem,
                                     device_gpu_timetable const tt) {
  if(!query.ontrip_) {
    reset_ea(device_mem.earliest_arrivals_, device_mem.arrival_times_count_);
    this_grid().sync();
  }

  auto const trait_size = CriteriaConfig::TRAITS_SIZE;
  auto const t_id = get_global_thread_id();

  mc_init_arrivals_dev<CriteriaConfig>(query, device_mem, tt);
  this_grid().sync();

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    if (t_id < trait_size) {
      device_mem.any_station_marked_[t_id] = false;
    }
    if (t_id == 0) {
      *(device_mem.overall_station_marked_) = false;
    }
    this_grid().sync();

    mc_convert_station_to_route_marks(
        device_mem.station_marks_, device_mem.route_marks_,
        device_mem.any_station_marked_, device_mem.overall_station_marked_, tt,
        trait_size);
    this_grid().sync();

    auto const station_store_size = ((tt.stop_count_ * trait_size) / 32) + 1;
    reset_store(device_mem.station_marks_, station_store_size);
    this_grid().sync();

    if (!(*device_mem.overall_station_marked_)) {
      return;
    }

    mc_update_routes_dev<CriteriaConfig>(device_mem, round_k, query.target_,
                                         tt);

    this_grid().sync();

    perform_arrival_sweeping<CriteriaConfig>(
        device_mem.stop_count_, device_mem.result_[round_k],
        device_mem.earliest_arrivals_, device_mem.station_marks_);
    this_grid().sync();

    mc_update_footpaths_dev<CriteriaConfig>(device_mem, round_k, query.target_,
                                            tt);
    this_grid().sync();

    if (t_id == 0) {
      device_mem.stats_->number_of_rounds_ = round_k;
    }
  }
}

template <typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const& dq) {
  void* kernel_args[] = {(void*)&dq, (void*)(dq.mem_->active_device_),
                         (void*)&dq.tt_
  };

  launch_kernel(mc_gpu_raptor_kernel<CriteriaConfig>, kernel_args,
                dq.mem_->context_, dq.mem_->context_.proc_stream_,
                dq.criteria_config_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.proc_stream_);
  cuda_check();

  fetch_arrivals_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();

  fetch_statistics_async(dq, dq.mem_->context_.transfer_stream_);
  cuda_check();

  cuda_sync_stream(dq.mem_->context_.transfer_stream_);
  cuda_check();
}

#define GENERATE_LAUNCH_CONFIG_FUNCTION(VAL, ACCESSOR)                        \
  template <>                                                                 \
  std::pair<dim3, dim3> get_mc_gpu_raptor_launch_parameters<VAL>(             \
      device_id const device_id, int32_t const concurrency_per_device) {      \
    cudaSetDevice(device_id);                                                 \
    cuda_check();                                                             \
                                                                              \
    cudaDeviceProp prop{};                                                    \
    cudaGetDeviceProperties(&prop, device_id);                                \
    cuda_check();                                                             \
                                                                              \
    utl::verify(prop.warpSize == 32,                                          \
                "Warp Size must be 32! Otherwise the gRAPTOR algorithm will " \
                "not work.");                                                 \
                                                                              \
    int min_grid_size = 0;                                                    \
    int block_size = 0;                                                       \
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,           \
                                       mc_gpu_raptor_kernel<VAL>, 0, 0);      \
                                                                              \
    dim3 threads_per_block(prop.warpSize, block_size / prop.warpSize, 1);     \
    dim3 grid(min_grid_size / concurrency_per_device, 1, 1);                  \
                                                                              \
    return {threads_per_block, grid};                                         \
  }

#define MAKE_MC_GPU_RAPTOR_TEMPLATE_INSTANCE(VAL, ACCESSOR) \
  template void invoke_mc_gpu_raptor<VAL>(const d_query& dq);

#define MAKE_MC_INIT_ARRIVALS_TEMPLATE_INSTANCE(VAL, ACCESSOR) \
  template __device__ void mc_init_arrivals_dev<VAL>(          \
      base_query const&, device_memory const&, device_gpu_timetable const&);

#define MAKE_MC_UPDATE_FOOTPATHS_TEMPLATE_INSTANCE(VAL, ACCESSOR) \
  template __device__ void mc_update_footpaths_dev<VAL>(          \
      device_memory const&, raptor_round const, stop_id const,    \
      device_gpu_timetable const&);

#define MAKE_MC_UPDATE_ROUTES_TEMPLATE_INSTANCE(VAL, ACCESSOR) \
  template __device__ void mc_update_routes_dev<VAL>(          \
      device_memory const&, raptor_round const, stop_id const, \
      device_gpu_timetable const&);

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(GENERATE_LAUNCH_CONFIG_FUNCTION,
                                   raptor_criteria_config)

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_INIT_ARRIVALS_TEMPLATE_INSTANCE, )
RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_UPDATE_FOOTPATHS_TEMPLATE_INSTANCE, )
RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_UPDATE_ROUTES_TEMPLATE_INSTANCE, )
RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_GPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor
