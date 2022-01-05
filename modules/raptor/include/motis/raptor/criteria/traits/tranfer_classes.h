#pragma once

#include <cstdint>
#include <algorithm>
#include <tuple>

#include "motis/raptor/raptor_util.h"
#include "motis/raptor/types.h"

#include "motis/core/journey/journey.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "cooperative_groups.h"
#endif

namespace motis::raptor {

struct device_gpu_timetable;

struct trait_max_transfer_class {

  // a lower class means slower transfer times
  // therefore we have
  //  0 => slow transfer
  //  1 => normal transfer
  //  2 => fast transfer
  static constexpr uint32_t _max_transfer_class = 2;

  static constexpr float fast_tt_multiplier = 0.7f;
  static constexpr float slow_tt_multiplier = 1.5f;

  // indices
  static constexpr uint32_t _index_range_size_ = _max_transfer_class + 1;

  // Trait Data
  transfer_class_t initial_max_transfer_class_{0};
  transfer_class_t max_transfer_class_{0};

  _mark_cuda_rel_ inline static dimension_id index_range_size() {
    return _index_range_size_;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static dimension_id get_write_to_dimension_id(
      TraitsData const& d) {
    return (d.max_transfer_class_ < d.initial_max_transfer_class_)
               ? d.initial_max_transfer_class_
               : d.max_transfer_class_;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx == get_write_to_dimension_id(data);
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx < data.max_transfer_class_;
  }

  inline static bool is_forward_propagation_required() {
    // we can find all valid solutions by individually checking each offset
    return false;
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static std::tuple<bool, bool> set_and_check_departure(
      TraitsData& aggregate, Timetable const& tt, stop_offset const dep_stop,
      time const prev_arrival, time const stop_departure) {
    stop_id const stop_id =
        tt.route_stops_[aggregate.route_->index_to_route_stops_ + dep_stop];

    time const regular_tt = tt.transfer_times_[stop_id];
    time const slow_tt    = slow_tt_multiplier * regular_tt;
    time const fast_tt    = fast_tt_multiplier * regular_tt;

    time const total_tt = stop_departure - prev_arrival;

    if (total_tt >= slow_tt) {
      aggregate.max_transfer_class_ = 0;
      return std::make_tuple(true, true);
    }else if(total_tt >= regular_tt) {
      aggregate.max_transfer_class_ = 1;
      return std::make_tuple(true, true);
    }else if(total_tt >= fast_tt) {
      aggregate.max_transfer_class_ = 2;
      return std::make_tuple(true, true);
    }

    return std::make_tuple(true, false);
  }

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(
      TraitsData& td, Timetable const& tt, stop_offset const current_stop,
      stop_times_index const current_sti) {
    // intentionally empty as updates per stop are not needed for this criteria
  }

#if defined(MOTIS_CUDA)

  template <typename TraitsData>
  __device__ inline static void propagate_and_merge_if_needed(
      TraitsData& aggregate, unsigned const mask, bool const is_departure_stop,
      bool const write_update) {
    // intentionally empty as updates per stop are not needed for this criteria
  }

  template <typename TraitsData>
  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    TraitsData& aggregate) {
    // on routes with more than 32 stops and a departure stop on the previous
    //  stage we need to carry over the value from the last stop to the first
    //  of the next stage
    auto const prop_val = aggregate.max_transfer_class_;
    auto const received = __shfl_down_sync(mask, prop_val, 31);
    aggregate.max_transfer_class_ = received;
  }

  template <typename TraitsData>
  __device__ inline static void calculate_aggregate(
      TraitsData& aggregate, device_gpu_timetable const& tt,
      stop_times_index const dep_sti, stop_times_index const arr_sti) {
    // intentionally empty as updates per stop are not needed for this criteria
  }

#endif

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void reset_aggregate(
      TraitsData& aggregate_dt, dimension_id const initial_dim_id) {
    aggregate_dt.max_transfer_class_ = invalid<transfer_class_t>;
    aggregate_dt.initial_max_transfer_class_ = initial_dim_id;
  }

  //****************************************************************************
  // below is used solely in reconstructor

  template <typename TraitsData>
  inline static std::vector<dimension_id> get_feasible_dimensions(
      dimension_id const initial_offset, TraitsData const&) {

    if (initial_offset == 2) return std::vector<dimension_id>{0, 1, 2};
    if (initial_offset == 1) return std::vector<dimension_id>{0, 1};
    if (initial_offset == 0) return std::vector<dimension_id>{0};

    return std::vector<dimension_id>{};
  }

  inline static bool dominates(dimension_id const to_dominate,
                               dimension_id const dominating) {
    return dominating <= to_dominate;
  }

  inline static void fill_journey(journey& j, dimension_id const dim) {
    j.max_transfer_class_ = dim;
  }

private:
  template <typename Timetable>
  _mark_cuda_rel_ static inline time _get_departure_time(
      Timetable const& tt, stop_times_index const departure_sti) {
    auto const& stop_times = tt.stop_times_[departure_sti];
    return stop_times.departure_;
  }
};

#if defined(MOTIS_CUDA)
template <>
_mark_cuda_rel_ inline time
trait_max_transfer_class::_get_departure_time<device_gpu_timetable>(
    device_gpu_timetable const& tt, stop_times_index const departure_sti) {
  return tt.stop_departures_[departure_sti];
}
#endif

}  // namespace motis::raptor
