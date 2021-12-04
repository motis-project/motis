#pragma once

#include <tuple>

#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"

namespace motis::raptor {

template <typename Traits>
struct criteria_config {
  using CriteriaData = typename Traits::TraitsData;

  __mark_cuda_rel__ inline static trait_id trait_size() {
    return Traits::size();
  }

  __mark_cuda_rel__ inline static int get_arrival_idx(
      stop_id const stop_idx, trait_id const trait_offset = 0) {
    return stop_idx * trait_size() + trait_offset;
  }

  //  // TODO get rid of
  //  __mark_cuda_rel__ inline static bool is_update_required(
  //      CriteriaData const& td, trait_id t_offset) {
  //    return Traits::is_update_required(trait_size(), td, t_offset);
  //  }

  __mark_cuda_rel__ inline static trait_id get_write_to_trait_id(
      CriteriaData& d) {
    return Traits::get_write_to_trait_id(d);
  }

  // TODO use again
  inline static bool is_trait_satisfied(CriteriaData const& td,
                                        trait_id t_offset) {
    return Traits::is_trait_satisfied(trait_size(), td, t_offset);
  }

  //****************************************************************************
  // Only used by CPU RAPTOR
  inline static bool is_rescan_from_stop_needed(CriteriaData const& td,
                                                trait_id t_offset) {
    return Traits::is_rescan_from_stop_needed(trait_size(), td, t_offset);
  }
  //****************************************************************************

  template <typename Timetable>
  __mark_cuda_rel__ inline static void update_traits_aggregate(
      CriteriaData& aggregate_dt, Timetable const& tt,
      time const* const prev_arrivals, stop_offset const s_offset,
      stop_times_index const current_sti) {
    Traits::update_aggregate(aggregate_dt, tt, prev_arrivals, s_offset,
                             current_sti, trait_size());
  }

  __mark_cuda_rel__ inline static void reset_traits_aggregate(
      CriteriaData& dt, route_id const r_id, trip_id const t_id,
      trait_id const initial_offset) {
    dt.route_id_ = r_id;
    dt.trip_id_ = t_id;
    dt.departure_trait_id_ = initial_offset;
    Traits::reset_aggregate(trait_size(), dt, initial_offset);
  }

#if defined(MOTIS_CUDA)

  __device__ inline static void propagate_and_merge_if_needed(
      unsigned const mask, CriteriaData& aggregate, bool const predicate) {
    Traits::propagate_and_merge_if_needed(mask, aggregate, predicate);
  }

  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    CriteriaData& aggregate) {
    Traits::carry_to_next_stage(mask, aggregate);
  }

#endif

  // derive the trait values from the arrival time index
  // expecting that the stop_idx is already subtracted and the given index
  // only specifies the shifts within the traits
  inline static CriteriaData get_traits_data(trait_id const trait_offset) {
    CriteriaData data{};
    Traits::get_trait_data(trait_size(), data, trait_offset);
    return data;
  }

  inline static std::vector<trait_id> get_feasible_traits(
      trait_id const initial_offset, CriteriaData const& new_trip) {
    auto feasible = Traits::get_feasible_trait_ids(trait_size(), initial_offset,
                                                   new_trip);
    std::reverse(std::begin(feasible), std::end(feasible));
    return feasible;
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  inline static bool dominates(CriteriaData const& to_dominate,
                               CriteriaData const& dominating) {
    return Traits::dominates(to_dominate, dominating);
  }
};

}  // namespace motis::raptor
