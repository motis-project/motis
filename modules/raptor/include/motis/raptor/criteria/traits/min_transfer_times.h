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

struct trait_min_transfer_times {

  // min transfer times is slotted into 5 minute intervals
  //  0, 5, 10, 15, 20 minutes required transfer time per trip change
  static constexpr uint8_t _min_transfer_time_max_val_ = 20;
  static constexpr uint8_t _min_transfer_time_max_idx_ = 4;

  // indices
  static constexpr uint32_t _index_range_size_ =
      _min_transfer_time_max_idx_ + 1;

  // Trait Data
  dimension_id initial_mtt_idx_{};
  uint8_t min_transfer_time_idx_{};

  _mark_cuda_rel_ inline static dimension_id index_range_size() {
    return _index_range_size_;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static dimension_id get_write_to_dimension_id(
      TraitsData const& d) {
    return (d.initial_mtt_idx_ < d.min_transfer_time_idx_)
               ? d.min_transfer_time_idx_
               : d.initial_mtt_idx_;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx == 0 && data.min_transfer_time_idx_ == 0;
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx > data.min_transfer_time_idx_;
  }

  inline static bool is_forward_propagation_required() {
    // we can find all valid solutions by individually checking each offset
    return false;
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(
      TraitsData& td, Timetable const& tt, time const* const previous_arrivals,
      stop_offset const current_stop, stop_times_index const current_sti,
      trait_id const total_trait_size) {
    // this method is called for all stops behind a departure station
    // every departure station resets the aggregate
    // therefore, we can do a look back and
    if (!valid(td.min_transfer_time_idx_)) {
      auto const departure_sti = current_sti - 1;
      auto const departure_time = _get_departure_time(tt, departure_sti);

      auto const& route = tt.routes_[td.route_id_];
      auto const dep_s_id =
          tt.route_stops_[route.index_to_route_stops_ + current_stop - 1];

      // skip the arrival index back to the departure station
      auto const departure_arr_idx =
          dep_s_id * total_trait_size + td.departure_trait_id_;
      auto const arrival_at_dep = previous_arrivals[departure_arr_idx];

      td.min_transfer_time_idx_ = (departure_time - arrival_at_dep) / 5;
      if (td.min_transfer_time_idx_ > 4) td.min_transfer_time_idx_ = 4;
    }
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
    // intentionally empty as updates per stop are not needed for this criteria
  }

#endif

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void reset_aggregate(
      TraitsData& aggregate_dt, dimension_id const initial_dim_id) {
    aggregate_dt.min_transfer_time_idx_ = invalid<uint8_t>;
    aggregate_dt.initial_mtt_idx_ = initial_dim_id;
  }

  //****************************************************************************
  // below is used solely in reconstructor

  template <typename TraitsData>
  inline static std::vector<dimension_id> get_feasible_dimensions(
      dimension_id const initial_offset, TraitsData const& data) {
    // TODO;
    return std::vector<dimension_id>{};
  }

  inline static bool dominates(dimension_id const to_dominate,
                               dimension_id const dominating) {
    // TODO
    return dominating <= to_dominate;
  }

  inline static void fill_journey(journey& j, dimension_id const dim) {
    j.min_transfer_time_ = dim;
  }

private:
  template <typename Timetable>
  _mark_cuda_rel_ static inline time _get_departure_time(
      Timetable const& tt, stop_times_index const departure_sti) {
    auto const& stop_times = tt.stop_times_[departure_sti];
    return stop_times.departure_;
  }

#if defined(MOTIS_CUDA)
  template <>
  _mark_cuda_rel_ static inline time _get_departure_time<device_gpu_timetable>(
      device_gpu_timetable const& tt, stop_times_index const departure_sti) {
    return tt.stop_departures_[departure_sti];
  }
#endif
};

}  // namespace motis::raptor
