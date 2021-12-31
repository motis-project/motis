#pragma once

#include <tuple>

#include "motis/raptor/raptor_util.h"
#include "motis/raptor/types.h"

namespace motis {
struct journey;
}

namespace motis::raptor {

enum class CalcMethod {
  Shfl,
  Flat
};

template <typename Traits, CalcMethod calc>
struct criteria_config {
  using CriteriaData = typename Traits::TraitsData;
  static constexpr auto UsesShflCalc = calc == CalcMethod::Shfl;

  _mark_cuda_rel_ inline static trait_id trait_size() { return Traits::size(); }

  _mark_cuda_rel_ inline static arrival_id get_arrival_idx(
      stop_id const stop_idx, trait_id const trait_offset = 0) {
    return stop_idx * trait_size() + trait_offset;
  }

  _mark_cuda_rel_ inline static trait_id get_write_to_trait_id(
      CriteriaData& d) {
    return Traits::get_write_to_trait_id(d);
  }

  _mark_cuda_rel_ inline static bool is_trait_satisfied(CriteriaData const& td,
                                                        trait_id t_offset) {
    return Traits::is_trait_satisfied(trait_size(), td, t_offset);
  }

  //****************************************************************************
  // Only used by CPU RAPTOR
  inline static bool is_rescan_from_stop_needed(CriteriaData const& td,
                                                trait_id t_offset) {
    return Traits::is_rescan_from_stop_needed(trait_size(), td, t_offset);
  }

  inline static bool is_forward_propagation_required() {
    return Traits::is_forward_propagation_required();
  }
  //****************************************************************************

  template <typename Timetable>
  _mark_cuda_rel_ inline static void update_traits_aggregate(
      CriteriaData& aggregate_dt, Timetable const& tt,
      stop_offset const s_offset, stop_times_index const current_sti) {
    Traits::update_aggregate(aggregate_dt, tt, s_offset, current_sti);
  }

  _mark_cuda_rel_ inline static void reset_traits_aggregate(
      CriteriaData& dt, route_id const r_id, trip_id const t_id,
      trait_id const initial_offset) {
    dt.route_id_ = r_id;
    dt.trip_id_ = t_id;
    dt.departure_trait_id_ = initial_offset;
    Traits::reset_aggregate(trait_size(), dt, initial_offset);
  }

#if defined(MOTIS_CUDA)

  __device__ inline static void propagate_and_merge_if_needed(
      unsigned const mask, CriteriaData& aggregate,
      bool const is_departure_stop, bool const write_value) {
    Traits::propagate_and_merge_if_needed(mask, aggregate, is_departure_stop,
                                          write_value);
  }

  template <typename Timetable>
  __device__ inline static void calculate_traits_aggregate(
      CriteriaData& aggregate, Timetable const& tt,
      stop_times_index const dep_sti, stop_times_index const arr_sti) {
    Traits::calculate_aggregate(aggregate, tt, dep_sti, arr_sti);
  }

  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    CriteriaData& aggregate) {
    Traits::carry_to_next_stage(mask, aggregate);
  }

#endif

  inline static std::vector<trait_id> get_feasible_traits(
      trait_id const initial_offset, CriteriaData const& new_trip) {
    auto feasible =
        Traits::get_feasible_trait_ids(trait_size(), initial_offset, new_trip);
    std::reverse(std::begin(feasible), std::end(feasible));
    return feasible;
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  inline static bool dominates(trait_id const to_dominate,
                               trait_id const dominating) {
    return Traits::dominates(trait_size(), to_dominate, dominating);
  }

  inline static void fill_journey(journey& journey, trait_id const t_offset) {
    Traits::fill_journey(trait_size(), journey, t_offset);
  }
};

}  // namespace motis::raptor
