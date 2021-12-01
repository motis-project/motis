#pragma once

#include <cstdint>
#include <algorithm>
#include <tuple>

#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "cooperative_groups.h"
#endif

namespace motis::raptor {

struct device_gpu_timetable;

struct trait_time_slotted_occupancy {

  /**
   * Time slotted occupancy trait implements a criteria which has a
   * time weighted occupancy measure broken down into discrete time slots.
   *
   * During route scanning we aggregate the duration of each segment by the
   * occupancy within this segment. When writing an arrival time a target
   * trait offset needs to be determined from the multiplication aggregate.
   * This is done by dividing the maximal possible weights occupancy value
   * into discrete slots. The target trait value can then be found by
   * checking in which slot the value fits.
   *
   * Currently the maximal duration of a connection is 1440 minutes (24 horus)
   * and the maximal occupancy possible on a trip segment is 2. This means that
   * the maximal reachable value is 2 occ * 1440 min => 2880 occupancy minutes.
   *
   * Then dividing 2880 into discrete segments can be done in different ways.
   * Either dividing it into equally sized slots, e.g. by dividing it by 45
   * we get 64 slots. A division by 45 would imply, that 22,5 minutes on a trip
   * with occupancy 2 is equal to 45 minutes on a trip with occupancy 1.
   *
   * One can also think of more sophisticated splits. When thinking about
   * common duration of a connection, it can be seen that a connection rarely
   * exceeds the 12 hour mark, at least when comparing national (within e.g.
   * Germany) connections. Therefore, a stages splitting can be employed.
   * The first 720 minutes * 2 occupancy => 1440 occupancy minutes could be
   * divided by 30 giving the first 46 slots and the remaining 1440 divided by
   * 60 giving the another 24 slots, yielding 46 + 24 => 70 slots in total.
   *
   * This implementation currently employs the first proposed model.
   */

  static constexpr uint32_t max_conn_duration = 1440;  // minutes
  static constexpr uint32_t max_occupancy_value = 2;
  static constexpr uint32_t max_time_occ_value =
      max_conn_duration * max_occupancy_value;  //^= 2880
  static constexpr uint32_t slot_divisor = 45;
  static constexpr uint32_t slot_count =
      max_time_occ_value / slot_divisor;  // 64

  // Trait Data
  dimension_id initial_soc_idx_{};
  uint32_t summed_occ_time_{};
  uint32_t occ_time_slot_{};

  __mark_cuda_rel__ inline static dimension_id index_range_size() {
    // slots match linearly to indices
    return slot_count;
  }

  // TODO check if this can be removed by the new method of determining a target
  //      occupancy level
  template <typename TraitsData>
  __mark_cuda_rel__ inline static bool is_update_required(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx == data.occ_time_slot_;
  }

  template <typename TraitsData>
  __mark_cuda_rel__ inline static dimension_id get_write_to_dimension_id(
      TraitsData const& d) {
    return d.initial_soc_idx_ + d.occ_time_slot_;
  }

  template <typename TraitsData>
  inline static bool is_trait_satisfied(TraitsData const& data,
                                        dimension_id const dimension_idx) {
    return dimension_idx == 0 && data.occ_time_slot_ == 0;  // TODO
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx < data.occ_time_slot_;  // TODO
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  __mark_cuda_rel__ inline static void update_aggregate(
      TraitsData& aggregate_dt, Timetable const& tt, time const* const _1,
      stop_offset const _2, stop_times_index const current_sti,
      trait_id const _3) {

    auto const stop_occupancy = _read_occupancy(tt, current_sti);
    auto const segment_duration = _read_segment_duration(tt, current_sti);
    aggregate_dt.summed_occ_time_ += (stop_occupancy * segment_duration);
    aggregate_dt.occ_time_slot_ = aggregate_dt.summed_occ_time_ / slot_divisor;
  }

  template <typename Timetable>
  __mark_cuda_rel__ inline static uint8_t _read_occupancy(
      Timetable const& tt, stop_times_index const sti) {
    return tt.stop_attr_[sti].inbound_occupancy_;
  }

  template <typename Timetable>
  __mark_cuda_rel__ inline static uint32_t _read_segment_duration(
      Timetable const& tt, stop_times_index const current_sti) {
    // because og stop time index alignment and the additional knowledge, that
    // we can't use a trip to arrive at the first stop we can safely reduce sti
    //  by one to get the time of the previous stop

    auto const previous_sti = current_sti - 1;

    // always use the segment duration and ignore stand times at a station
    auto const& previous_times = tt.stop_times_[previous_sti];
    auto const& current_times = tt.stop_times_[current_sti];

    return current_times.arrival_ - previous_times.departure_;
  }

#if defined(MOTIS_CUDA)
  uint32_t _segment_prop_occ_time_ = invalid<uint32_t>;

  template <>
  __mark_cuda_rel__ inline static uint8_t _read_occupancy<device_gpu_timetable>(
      device_gpu_timetable const& tt, stop_times_index const sti) {
    return tt.stop_inb_occupancy_[sti];
  }

  template <>
  __mark_cuda_rel__ inline static uint32_t
  _read_segment_duration<device_gpu_timetable>(device_gpu_timetable const& tt,
                                               stop_times_index const sti) {
    auto const prev_sti = sti - 1;
    auto const departure_time = tt.stop_departures_[prev_sti];
    auto const arrival_time = tt.stop_arrivals_[sti];
    return arrival_time - departure_time;
  }

  template <typename TraitsData>
  __device__ inline static void propagate_and_merge_if_needed(
      TraitsData& aggregate, unsigned const mask, bool const predicate) {
    if (!valid(aggregate._segment_prop_occ_time_)) {
      // there is always a call to update before the propagation is done
      // store this value to always repeat the same value to the next one
      aggregate._segment_prop_occ_time_ = aggregate.summed_occ_time_;
    }
    auto const prop_val = aggregate.summed_occ_time_;
    auto const received = __shfl_up_sync(mask, prop_val, 1);
    if (predicate) {
      aggregate.summed_occ_time_ = received + aggregate._segment_prop_occ_time_;
      aggregate.occ_time_slot_ = aggregate.summed_occ_time_ / slot_divisor;
    }
  }

  template <typename TraitsData>
  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    TraitsData& aggregate) {
    auto const prop_val = aggregate.summed_occ_time_;
    auto const received = __shfl_down_sync(mask, prop_val, 31);
    aggregate.summed_occ_time_ = received;
    aggregate._segment_prop_occ_time_ = invalid<uint32_t>;
  }

#endif

  template <typename TraitsData>
  __mark_cuda_rel__ inline static void reset_aggregate(
      TraitsData& aggregate_dt, dimension_id const initial_dim_id) {
    aggregate_dt.summed_occ_time_ = 0;
    aggregate_dt.occ_time_slot_ = 0;
    aggregate_dt.initial_soc_idx_ = initial_dim_id;

#if defined(MOTIS_CUDA)
    aggregate_dt._segment_prop_occ_time_ = invalid<uint32_t>;
#endif
  }

  //****************************************************************************
  // below is used solely in reconstructor

  template <typename TraitsData>
  inline static void fill_trait_data_from_idx(TraitsData& dt,
                                              uint32_t const dimension_idx) {
    // can be used as occupancy at idx 0
    //  maps to an occupancy value of 0
    dt.occ_time_slot_ = dimension_idx;

    // when determined this way only gives a lower bound not the actual value
    dt.summed_occ_time_ = dimension_idx * slot_divisor;
  }

  template <typename TraitsData>
  static bool dominates(TraitsData const& to_dominate,
                        TraitsData const& dominating) {
    return dominating.occ_time_slot_ <= to_dominate.occ_time_slot_;
  }
};

}  // namespace motis::raptor