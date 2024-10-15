#pragma once

#include <cstdint>
#include <algorithm>
#include <tuple>

#include "motis/raptor/raptor_util.h"
#include "motis/raptor/types.h"

#include "motis/core/journey/journey.h"
#include "utl/verify.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "cooperative_groups.h"
#include "cuda_runtime.h"
#endif

namespace motis::raptor {

template <typename Timetable>
_mark_cuda_rel_ inline static occ_t _read_occupancy(
    Timetable const& tt, stop_times_index const sti) {
  return tt.stop_attr_[sti].inbound_occupancy_;
}

template <typename Timetable>
_mark_cuda_rel_ inline static time _read_segment_duration(
    Timetable const& tt, stop_times_index const current_sti) {
  // because og stop time index alignment and the additional knowledge, that
  // we can't use a trip to arrive at the first stop we can safely reduce sti
  //  by one to get the time of the previous stop

  auto const previous_sti = current_sti - 1;

  // always use the segment duration and ignore stand times at a station
  auto const& previous_times = tt.stop_times_[previous_sti];
  auto const& current_times = tt.stop_times_[current_sti];

  auto const dep_time = valid(previous_times.departure_)
                            ? previous_times.departure_
                            : previous_times.arrival_;
  auto const arr_time = valid(current_times.arrival_)
                            ? current_times.arrival_
                            : current_times.departure_;

  return arr_time - dep_time;
}

struct data_time_slotted_occupancy {
  dimension_id initial_soc_idx_{};
  uint32_t summed_occ_time_{};
  uint32_t occ_time_slot_{};
#if defined(MOTIS_CUDA)
  uint32_t _segment_prop_occ_time_ = invalid<uint32_t>;
#endif
};

template <dimension_id SlotCount>
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

  static constexpr uint32_t slot_divisor = max_time_occ_value / SlotCount;

  constexpr static dimension_id DIMENSION_SIZE = SlotCount;
  constexpr static bool REQ_DIMENSION_PROPAGATION = false;
  constexpr static bool CHECKS_TRANSFER_TIME = false;

  // Trait Data
  dimension_id initial_soc_idx_{};
  uint32_t summed_occ_time_{};
  uint32_t occ_time_slot_{};

  template <typename TraitsData>
  _mark_cuda_rel_ inline static dimension_id get_write_to_dimension_id(
      TraitsData const& d) {
    int const write_to = d.initial_soc_idx_ + d.occ_time_slot_;
    return write_to < SlotCount ? static_cast<dimension_id>(write_to)
                                : invalid<dimension_id>;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx == get_write_to_dimension_id(data);
  }

  template <typename Timetable, typename TraitsData>
  _mark_cuda_rel_ inline static bool check_departure_stop(
      Timetable const&, TraitsData&, stop_offset const, stop_id const,
      time const, time const) {
    return true;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void set_departure_stop(TraitsData& aggregate,
                                                        stop_offset const) {
    aggregate.summed_occ_time_ = 0;
    aggregate.occ_time_slot_ = 0;
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx < data.occ_time_slot_;
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(
      TraitsData& aggregate_dt, Timetable const& tt, stop_offset const,
      stop_times_index const current_sti) {

    auto const stop_occupancy = _read_occupancy(tt, current_sti);
    auto const segment_duration = _read_segment_duration(tt, current_sti);

    aggregate_dt.summed_occ_time_ += (stop_occupancy * segment_duration);
    aggregate_dt.occ_time_slot_ = aggregate_dt.summed_occ_time_ / slot_divisor;

#if defined(MOTIS_CUDA)
    // only update the segment prop value on the first update after reset
    if (!valid(aggregate_dt._segment_prop_occ_time_))
      aggregate_dt._segment_prop_occ_time_ =
          (stop_occupancy * segment_duration);
#endif
  }

#if defined(MOTIS_CUDA)
  template <typename TraitsData>
  __device__ inline static void propagate_and_merge_if_needed(
      TraitsData& aggregate, unsigned const mask, bool const is_departure_stop,
      bool const write_update) {
    //    if (valid(aggregate._segment_prop_occ_time_)) {
    //      // there is always a call to update before the propagation is done
    //      // store this value to always repeat the same value to the next one
    //      aggregate.summed_occ_time_ = aggregate._segment_prop_occ_time_;
    //    }
    auto const prop_val = is_departure_stop ? 0 : aggregate.summed_occ_time_;
    auto const received = __shfl_up_sync(mask, prop_val, 1);
    if (write_update) {
      aggregate.summed_occ_time_ = received + aggregate._segment_prop_occ_time_;
      aggregate.occ_time_slot_ = aggregate.summed_occ_time_ / slot_divisor;
    }
  }

  template <typename TraitsData>
  __device__ inline static void calculate_aggregate(
      TraitsData& aggregate, device_gpu_timetable const& tt,
      time const* const,
      stop_times_index const dep_sti, stop_times_index const arr_sti) {

    for (stop_times_index current = dep_sti + 1; current <= arr_sti;
         ++current) {
      auto const duration = _read_segment_duration(tt, current);
      auto const occupancy = _read_occupancy(tt, current);
      aggregate.summed_occ_time_ += (duration * occupancy);
    }
    aggregate.occ_time_slot_ = aggregate.summed_occ_time_ / slot_divisor;
  }

  template <typename TraitsData>
  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    TraitsData& aggregate) {
    auto const prop_val = aggregate.summed_occ_time_;
    auto const received = __shfl_down_sync(mask, prop_val, 31);
    aggregate.summed_occ_time_ = received;
    aggregate.occ_time_slot_ = aggregate.summed_occ_time_ / slot_divisor;
    aggregate._segment_prop_occ_time_ = invalid<uint32_t>;
  }

#endif

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void reset_aggregate(
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
  inline static std::vector<dimension_id> get_feasible_dimensions(
      dimension_id const initial_offset, TraitsData const& data) {

    // there is exactly one feasible dimension, which is the
    //  initial - what is consumed by the trip
    int const new_dimension = initial_offset - data.occ_time_slot_;
    if (new_dimension >= 0)
      return std::vector<dimension_id>{
          static_cast<dimension_id>(new_dimension)};

    return std::vector<dimension_id>{};
  }

  inline static bool dominates(dimension_id const to_dominate,
                               dimension_id const dominating) {
    return dominating <= to_dominate;
  }

  inline static void fill_journey(journey& j, dimension_id const dim) {
    j.time_slotted_occupancy_ = dim;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void fill_aggregate(TraitsData& d,
                                                    dimension_id const dim) {
    d.occ_time_slot_ = dim;
  }
};

#if defined(MOTIS_CUDA)
template <>
_mark_cuda_rel_ inline occ_t _read_occupancy<device_gpu_timetable>(
    device_gpu_timetable const& tt, stop_times_index const sti) {
  return tt.stop_inb_occupancy_[sti];
}

template <>
_mark_cuda_rel_ inline time _read_segment_duration<device_gpu_timetable>(
    device_gpu_timetable const& tt, stop_times_index const sti) {
  auto const prev_sti = sti - 1;
  auto departure_time = tt.stop_departures_[prev_sti];
  if (!valid(departure_time)) departure_time = tt.stop_arrivals_[prev_sti];

  auto arrival_time = tt.stop_arrivals_[sti];
  if (!valid(arrival_time)) arrival_time = tt.stop_departures_[sti];

  return arrival_time - departure_time;
}
#endif

}  // namespace motis::raptor
