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

struct data_max_transfer_class {
  transfer_class_t initial_transfer_class_{invalid<transfer_class_t>};
  transfer_class_t active_transfer_class_{invalid<transfer_class_t>};
  transfer_class_t _staged_transfer_class_{invalid<transfer_class_t>};
};

/***
 * Transfer classes use different transfer times to determine whether
 * a stop can serve as departure stop. A reverse ordering is used, so that
 * the lowest dimension value requires the most transfer time and therefore,
 * leads to having the higher arrival times in the lower dimension indices.
 * This prevents them from being dominated.
 *
 * If this criteria is used in conjunction with other criteria which rely
 * on the departure station to determine their respective dimension indices.
 * This is e.g. true for the Occupancy based criteria. To demonstrate the
 * problem examine the following example: A trip with 6 stops is scanned.
 * The stops with index 0 and 3 are feasible departure stops, but on stop with
 * index 3 only a fast transfer enable catching the train, where as on the first
 * stop the train can be reached even with a slow transfer. When having both
 * stops as valid departure stops for the occupancy criteria, the stops with
 * indices 4 and 5 calculate their value starting from stop 3 instead of 0.
 * Therefore, separate scans per dimension index are necessary, when such
 * criteria are present in the configuration. In this case use the parameter
 * to control the behaviour of this criteria.
 *
 * @tparam UsedWithDepartureRelying
 */
template <bool UsedWithDepartureRelying>
struct trait_max_transfer_class {

  // a lower class means slower transfer times
  // therefore we have
  //  0 => slow transfer
  //  1 => normal transfer
  //  2 => fast transfer
  static constexpr uint32_t _max_transfer_class = 2;

  // indices
  constexpr static dimension_id DIMENSION_SIZE = _max_transfer_class + 1;
  constexpr static bool REQ_DIMENSION_PROPAGATION = UsedWithDepartureRelying;
  constexpr static bool CHECKS_TRANSFER_TIME = true;

  static constexpr float fast_tt_multiplier = 0.7f;
  static constexpr float slow_tt_multiplier = 1.5f;

  // Trait Data
  transfer_class_t initial_transfer_class_{invalid<transfer_class_t>};
  transfer_class_t active_transfer_class_{invalid<transfer_class_t>};
  transfer_class_t _staged_transfer_class_{invalid<transfer_class_t>};

  template <typename TraitsData>
  _mark_cuda_rel_ inline static dimension_id get_write_to_dimension_id(
      TraitsData const& d) {
    return (d.active_transfer_class_ < d.initial_transfer_class_)
               ? d.initial_transfer_class_
               : d.active_transfer_class_;
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx == get_write_to_dimension_id(data);
  }

  template <typename Timetable, typename TraitsData>
  _mark_cuda_rel_ inline static bool check_departure_stop(
      Timetable const& tt, TraitsData& aggregate, stop_offset const,
      stop_id const dep_stop_id, time const prev_arrival,
      time const stop_departure) {
    time const regular_tt = tt.transfer_times_[dep_stop_id];
    time const slow_tt = slow_tt_multiplier * regular_tt;
    time const fast_tt = fast_tt_multiplier * regular_tt;

    if (prev_arrival + slow_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 0;
      return true;
    } else if ((aggregate.initial_transfer_class_ >= 1 ||
                !UsedWithDepartureRelying) &&
               prev_arrival + regular_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 1;
      return true;
    } else if ((aggregate.initial_transfer_class_ >= 2 ||
                !UsedWithDepartureRelying) &&
               prev_arrival + fast_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 2;
      return true;
    } else {
      return false;
    }
  }

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void set_departure_stop(TraitsData& aggregate,
                                                        stop_offset const) {
    if (valid(aggregate._staged_transfer_class_)) {
      aggregate.active_transfer_class_ = aggregate._staged_transfer_class_;
      aggregate._staged_transfer_class_ = invalid<transfer_class_t>;
    }
  }

  //****************************************************************************
  // Used only in CPU based
  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(
      TraitsData const& data, dimension_id const dimension_idx) {
    return dimension_idx < data.active_transfer_class_;
  }
  //****************************************************************************

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(TraitsData&,
                                                      Timetable const&,
                                                      stop_offset const,
                                                      stop_times_index const) {
    // intentionally empty
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

  template <typename TraitsData>
  __device__ inline static void calculate_aggregate(
      TraitsData& aggregate, device_gpu_timetable const& tt,
      time const prev_arrival,
      stop_times_index const dep_sti, stop_times_index const arr_sti) {

    auto const first_sti = aggregate.route_->index_to_stop_times_ +
                           aggregate.route_->stop_count_ * aggregate.trip_id_;
    auto const dep_offset = dep_sti - first_sti;
    auto const dep_s_id = aggregate.route_->index_to_route_stops_ + dep_offset;

    auto const regular_tt = tt.transfer_times_[dep_s_id];
    auto const stop_departure = tt.stop_departures_[dep_sti];

    auto const slow_tt = regular_tt * slow_tt_multiplier;
    auto const fast_tt = regular_tt * fast_tt_multiplier;

    //when reaching this point we already checked on the thread responsible for
    //the given departure stop, therefore, we know that one of the categories
    //will fit, and we are only interested to find out which
    if (prev_arrival + slow_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 0;
    } else if (prev_arrival + regular_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 1;
    } else if (prev_arrival + fast_tt <= stop_departure) {
      aggregate._staged_transfer_class_ = 2;
    }
  }

#endif

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void reset_aggregate(
      TraitsData& aggregate_dt, dimension_id const initial_dim_id) {
    aggregate_dt._staged_transfer_class_ = invalid<transfer_class_t>;
    aggregate_dt.active_transfer_class_ = invalid<transfer_class_t>;
    aggregate_dt.initial_transfer_class_ = initial_dim_id;
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

  template <typename TraitsData>
  _mark_cuda_rel_ inline static void fill_aggregate(TraitsData& d,
                                                    dimension_id const dim) {
    d.active_transfer_class_ = dim;
  }
};

struct transfer_class_transfer_time_calculator {
  static constexpr float fast_tt_multiplier = 0.7f;
  static constexpr float slow_tt_multiplier = 1.5f;

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static time get_transfer_time(TraitsData const& data,
                                                       Timetable const& tt,
                                                       stop_id const s_id) {
    if (!valid(s_id)) return invalid<time>;

    auto const regular_tt = tt.transfer_times_[s_id];
    if (data.active_transfer_class_ == 0)
      return slow_tt_multiplier * regular_tt;
    if (data.active_transfer_class_ == 1) return regular_tt;
    if (data.active_transfer_class_ == 2)
      return fast_tt_multiplier * regular_tt;

    return invalid<time>;
  }
};

}  // namespace motis::raptor
