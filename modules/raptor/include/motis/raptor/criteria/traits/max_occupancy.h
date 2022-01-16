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

struct data_max_occupancy {
  dimension_id initial_moc_idx_{0};
  occ_t max_occupancy_{0};
};

struct trait_max_occupancy {
  constexpr static occ_t max_occupancy = 2;

  constexpr static dimension_id DIMENSION_SIZE = max_occupancy + 1;
  constexpr static bool REQ_DIMENSION_PROPAGATION = false;
  constexpr static bool CHECKS_TRANSFER_TIME = false;

  // Trait Data
  dimension_id initial_moc_idx_{};
  occ_t max_occupancy_{};

  template <typename TraitsData>
  _mark_cuda_rel_ inline static uint32_t get_write_to_dimension_id(
      TraitsData const& d) {
    if (d.initial_moc_idx_ < d.max_occupancy_) {
      return d.max_occupancy_;
    }
    return d.initial_moc_idx_;
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
    aggregate.max_occupancy_ = 0;
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx < data.max_occupancy_;
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(
      TraitsData& aggregate_dt, Timetable const& tt, stop_offset const,
      stop_times_index const current_sti) {

    auto const stop_occupancy = _read_occupancy(tt, current_sti);
    aggregate_dt.max_occupancy_ =
        std::max(aggregate_dt.max_occupancy_, stop_occupancy);
  }

#if defined(MOTIS_CUDA)
  template <typename TraitsData>
  __device__ inline static void propagate_and_merge_if_needed(
      TraitsData& aggregate, unsigned const mask, bool const is_departure_stop,
      bool const write_update) {
    auto const prop_val = is_departure_stop ? 0 : aggregate.max_occupancy_;
    auto const received = __shfl_up_sync(mask, prop_val, 1);
    if (write_update && aggregate.max_occupancy_ < received)
      aggregate.max_occupancy_ = received;
  }

  template <typename TraitsData>
  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    TraitsData& aggregate) {
    auto const prop_val = aggregate.max_occupancy_;
    auto const received = __shfl_down_sync(mask, prop_val, 31);
    aggregate.max_occupancy_ = received;
  }

  template <typename TraitsData>
  __device__ inline static void calculate_aggregate(
      TraitsData& aggregate, device_gpu_timetable const& tt,
      time const* const,
      stop_times_index const dep_sti, stop_times_index const arr_sti) {

    for (stop_times_index current = dep_sti + 1; current <= arr_sti;
         ++current) {
      auto const occupancy = _read_occupancy(tt, current);
      if (occupancy > aggregate.max_occupancy_)
        aggregate.max_occupancy_ = occupancy;
    }
  }
#endif

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void reset_aggregate(
      TraitsData& aggregate_dt, dimension_id const initial_dim_id) {
    aggregate_dt.max_occupancy_ = 0;
    aggregate_dt.initial_moc_idx_ = initial_dim_id;
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

  static bool dominates(dimension_id const to_dominate,
                        dimension_id const dominating) {
    return dominating <= to_dominate;
  }

  inline static void fill_journey(journey& j, dimension_id const dim) {
    j.max_occupancy_ = dim;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void fill_aggregate(TraitsData& d,
                                                    dimension_id const dim) {
    d.max_occupancy_ = dim;
  }

private:
  template <typename Timetable>
  _mark_cuda_rel_ inline static occ_t _read_occupancy(
      Timetable const& tt, stop_times_index const sti) {
    return tt.stop_attr_[sti].inbound_occupancy_;
  }
};

#if defined(MOTIS_CUDA)
template <>
_mark_cuda_rel_ inline occ_t
trait_max_occupancy::_read_occupancy<device_gpu_timetable>(
    device_gpu_timetable const& tt, stop_times_index const sti) {
  return tt.stop_inb_occupancy_[sti];
}
#endif

}  // namespace motis::raptor
