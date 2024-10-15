#pragma once

#include <tuple>
#include <type_traits>
#include <vector>

#include "utl/verify.h"

#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"
#include "motis/raptor/types.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_mark_store.cuh"
#include "cooperative_groups.h"
#endif

namespace motis {
struct journey;
}

namespace motis::raptor {

template <typename... Trait>
struct traits;

template <typename FirstTrait, typename... RestTraits>
struct traits<FirstTrait, RestTraits...> {

  constexpr static trait_id SIZE =
      FirstTrait::DIMENSION_SIZE * traits<RestTraits...>::SIZE;
  constexpr static bool REQ_DIMENSION_PROPAGATION =
      FirstTrait::REQ_DIMENSION_PROPAGATION ||
      traits<RestTraits...>::REQ_DIMENSION_PROPAGATION;

  constexpr static trait_id SWEEP_BLOCK_SIZE =
      (FirstTrait::REQ_DIMENSION_PROPAGATION ? 1 : FirstTrait::DIMENSION_SIZE) *
      traits<RestTraits...>::SWEEP_BLOCK_SIZE;

  static_assert(
      FirstTrait::REQ_DIMENSION_PROPAGATION ||
          (!FirstTrait::REQ_DIMENSION_PROPAGATION &&
           !traits<RestTraits...>::REQ_DIMENSION_PROPAGATION),
      "A trait without dimension propagation is not allowed on top of a trait "
      "with dimension propagation! Change the trait order accordingly.");

#if defined(MOTIS_CUDA)
  __device__ inline static void perform_stop_arrival_sweeping_gpu(
      arrival_id const total_trait_size, stop_id const s_id,
      time* const arrivals, uint32_t* station_marks) {
    // start with the innermost criterion
    traits<RestTraits...>::perform_stop_arrival_sweeping_gpu(
        total_trait_size, s_id, arrivals, station_marks);

    /**
     * Arrival sweeping is skipped for criteria which require that their
     * arrival times are propagated across the dimension, i.e. when influencing
     * the departure locations
     */
    if (REQ_DIMENSION_PROPAGATION) return;

    // now process the first trait criterion
    auto const c_phi_max = traits<RestTraits...>::SIZE;
    for (int offset = 0; offset < c_phi_max; ++offset) {
      time time_min = invalid<time>;
      arrival_id trait_offset = 0;
      int idx = 0;
      do {
        auto const arrival_id = s_id * total_trait_size + trait_offset;
        if (valid(time_min) && valid(arrivals[arrival_id]) &&
            time_min <= arrivals[arrival_id]) {
          arrivals[arrival_id] = invalid<time>;
          unmark(station_marks, arrival_id);
        } else if (valid(arrivals[arrival_id])) {
          time_min = arrivals[arrival_id];
        }

        ++idx;
        trait_offset = idx * c_phi_max + offset;

        if (idx % FirstTrait::DIMENSION_SIZE == 0)
          time_min = arrivals[arrival_id];

      } while (trait_offset < total_trait_size);
    }
  }
#endif

  inline static void perform_stop_arrival_sweeping_cpu(
      arrival_id const total_trait_size, stop_id const s_id,
      time* const arrivals, cpu_mark_store& station_marks) {
    // start with the innermost criterion
    traits<RestTraits...>::perform_stop_arrival_sweeping_cpu(
        total_trait_size, s_id, arrivals, station_marks);

    /**
     * Arrival sweeping is skipped for criteria which require that their
     * arrival times are propagated across the dimension, i.e. when influencing
     * the departure locations
     */
    if (REQ_DIMENSION_PROPAGATION) return;

    // now process the first trait criterion
    auto const c_phi_max = traits<RestTraits...>::SIZE;
    for (int offset = 0; offset < c_phi_max; ++offset) {
      time time_min = invalid<time>;
      arrival_id trait_offset = 0;
      int idx = 0;
      do {
        auto const arrival_id = s_id * total_trait_size + trait_offset;
        if (valid(time_min) && valid(arrivals[arrival_id]) &&
            time_min <= arrivals[arrival_id]) {
          arrivals[arrival_id] = invalid<time>;
          station_marks.unmark(arrival_id);
        } else if (valid(arrivals[arrival_id])) {
          time_min = arrivals[arrival_id];
        }

        ++idx;
        trait_offset = idx * c_phi_max + offset;

        if (idx % FirstTrait::DIMENSION_SIZE == 0)
          time_min = arrivals[arrival_id];

      } while (trait_offset < total_trait_size);
    }
  }

  template <typename CriteriaData>
  _mark_cuda_rel_ inline static trait_id get_write_to_trait_id(
      CriteriaData& d) {
    auto const first_dimension_idx = FirstTrait::get_write_to_dimension_id(d);
    if (!valid(first_dimension_idx)) {
      return invalid<trait_id>;
    }

    auto const first_dim_step_width = traits<RestTraits...>::SIZE;

    auto const rest_trait_offset =
        traits<RestTraits...>::get_write_to_trait_id(d);

    auto write_idx =
        first_dim_step_width * first_dimension_idx + rest_trait_offset;
    return write_idx;
  }

  template <typename CriteriaData>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(trait_id total_size,
                                                        CriteriaData const& td,
                                                        trait_id t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_trait_satisfied(td, first_dimension_idx) &&
           traits<RestTraits...>::is_trait_satisfied(rest_trait_size, td,
                                                     rest_trait_offset);
  }

  template <typename Timetable, typename CriteriaData>
  _mark_cuda_rel_ inline static bool check_departure_stop(
      bool checked_tt, Timetable const& tt, CriteriaData& aggregate,
      stop_offset const dep_offset, stop_id const dep_s_id,
      time const prev_arrival, time const stop_departure) {

    auto first_feasible = FirstTrait::check_departure_stop(
        tt, aggregate, dep_offset, dep_s_id, prev_arrival, stop_departure);
    return first_feasible &&
           traits<RestTraits...>::check_departure_stop(
               checked_tt || FirstTrait::CHECKS_TRANSFER_TIME, tt, aggregate,
               dep_offset, dep_s_id, prev_arrival, stop_departure);
  }

  template <typename CriteriaData>
  _mark_cuda_rel_ inline static void set_departure_stop(
      CriteriaData& data, stop_offset const dep_offset) {
    FirstTrait::set_departure_stop(data, dep_offset);
    traits<RestTraits...>::set_departure_stop(data, dep_offset);
  }

  //****************************************************************************
  // Only used by CPU RAPTOR
  template <typename CriteriaData>
  inline static bool is_rescan_from_stop_needed(trait_id total_size,
                                                CriteriaData const& td,
                                                trait_id t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_rescan_from_stop_needed(td, first_dimension_idx) ||
           traits<RestTraits...>::is_rescan_from_stop_needed(
               rest_trait_size, td, rest_trait_offset);
  }
  //****************************************************************************

  // helper to aggregate values while progressing through the route stop by stop
  template <typename Timetable, typename CriteriaData>
  _mark_cuda_rel_ inline static void update_aggregate(
      CriteriaData& aggregate_dt, Timetable const& tt,
      stop_offset const s_offset, stop_times_index const current_sti) {

    FirstTrait::update_aggregate(aggregate_dt, tt, s_offset, current_sti);

    traits<RestTraits...>::update_aggregate(aggregate_dt, tt, s_offset,
                                            current_sti);
  }

  // reset the aggregate everytime the departure station changes
  template <typename CriteriaData>
  _mark_cuda_rel_ inline static void reset_aggregate(
      trait_id const total_size, CriteriaData& aggregate_dt,
      trait_id const initial_t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, initial_t_offset);

    FirstTrait::reset_aggregate(aggregate_dt, first_dimension_idx);
    traits<RestTraits...>::reset_aggregate(rest_trait_size, aggregate_dt,
                                           rest_trait_offset);
  }

#if defined(MOTIS_CUDA)
  template <typename CriteriaData>
  __device__ inline static void propagate_and_merge_if_needed(
      unsigned const mask, CriteriaData& aggregate,
      bool const is_departure_stop, bool const write_update) {
    FirstTrait::propagate_and_merge_if_needed(aggregate, mask,
                                              is_departure_stop, write_update);
    traits<RestTraits...>::propagate_and_merge_if_needed(
        mask, aggregate, is_departure_stop, write_update);
  }

  template <typename CriteriaData>
  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    CriteriaData& aggregate) {
    FirstTrait::carry_to_next_stage(mask, aggregate);
    traits<RestTraits...>::carry_to_next_stage(mask, aggregate);
  }

  template <typename Timetable, typename CriteriaData>
  __device__ inline static void calculate_aggregate(
      CriteriaData& aggregate, Timetable const& tt,
      time const* const prev_arrivals, stop_times_index const dep_sti,
      stop_times_index const arr_sti) {

    FirstTrait::calculate_aggregate(aggregate, tt, prev_arrivals, dep_sti,
                                    arr_sti);
    traits<RestTraits...>::calculate_aggregate(aggregate, tt, prev_arrivals,
                                               dep_sti, arr_sti);
  }
#endif

  template <typename CriteriaData>
  inline static std::vector<trait_id> get_feasible_trait_ids(
      trait_id const total_size, trait_id const initial_offset,
      CriteriaData const& input) {

    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, initial_offset);

    auto const first_dimensions =
        FirstTrait::get_feasible_dimensions(first_dimension_idx, input);
    auto const first_size = traits<RestTraits...>::SIZE;
    auto const rest_feasible = traits<RestTraits...>::get_feasible_trait_ids(
        rest_trait_size, rest_trait_offset, input);

    std::vector<trait_id> total_feasible(first_dimensions.size() *
                                         rest_feasible.size());

    auto idx = 0;
    for (auto const first_dim : first_dimensions) {
      for (auto const rest_offset : rest_feasible) {
        total_feasible[idx] = first_dim * first_size + rest_offset;
        ++idx;
      }
    }

    return total_feasible;
  }

  inline static bool dominates(trait_id const total_size,
                               trait_id const to_dominate,
                               trait_id const dominating) {
    auto const [rest_size_todom, first_dim_todom, rest_trait_todom] =
        _trait_values(total_size, to_dominate);

    auto const [rest_size_domin, first_dim_domin, rest_trait_domin] =
        _trait_values(total_size, dominating);

    utl::verify_ex(rest_size_todom == rest_size_domin,
                   "Trait derivation failed!");

    return FirstTrait::dominates(first_dim_todom, first_dim_domin) &&
           traits<RestTraits...>::dominates(rest_size_domin, rest_trait_todom,
                                            rest_trait_domin);
  }

  inline static void fill_journey(trait_id const total_size, journey& journey,
                                  trait_id const t_offset) {
    auto const [rest_size, first_dimension, rest_offset] =
        _trait_values(total_size, t_offset);
    FirstTrait::fill_journey(journey, first_dimension);
    traits<RestTraits...>::fill_journey(rest_size, journey, rest_offset);
  }

  template <typename CriteriaData>
  _mark_cuda_rel_ inline static void fill_aggregate(trait_id const total_size,
                                                    CriteriaData& aggregate,
                                                    trait_id const t_offset) {
    auto const [rest_size, first_dimension, rest_offset] =
        _trait_values(total_size, t_offset);
    FirstTrait::fill_aggregate(aggregate, first_dimension);
    traits<RestTraits...>::fill_aggregate(rest_size, aggregate, rest_offset);
  }

private:
  _mark_cuda_rel_ inline static std::tuple<trait_id, dimension_id, trait_id>
  _trait_values(trait_id const total_size, trait_id const t_offset) {
    auto const first_value_size = FirstTrait::DIMENSION_SIZE;
    auto const rest_trait_size = total_size / first_value_size;

    auto const first_dimension_idx = t_offset / rest_trait_size;
    auto const rest_trait_offset = t_offset % rest_trait_size;

    return std::make_tuple(rest_trait_size, first_dimension_idx,
                           rest_trait_offset);
  }
};

template <>
struct traits<> {
  constexpr static trait_id SIZE = 1;
  constexpr static bool REQ_DIMENSION_PROPAGATION = false;
  constexpr static trait_id SWEEP_BLOCK_SIZE = 1;

#if defined(MOTIS_CUDA)
  __device__ inline static void perform_stop_arrival_sweeping_gpu(
      arrival_id const, stop_id const, time* const, uint32_t*) {}
#endif

  inline static void perform_stop_arrival_sweeping_cpu(arrival_id const,
                                                  stop_id const, time* const,
                                                  cpu_mark_store&) {}

  template <typename Data>
  _mark_cuda_rel_ inline static trait_id get_write_to_trait_id(Data const&) {
    return 0;
  }

  template <typename Data>
  _mark_cuda_rel_ inline static bool is_trait_satisfied(uint32_t, Data const&,
                                                        uint32_t) {
    return true;  // return natural element of conjunction
  }

  template <typename Data, typename Timetable>
  _mark_cuda_rel_ inline static bool check_departure_stop(
      bool checked_tt, Timetable const& tt, Data&, stop_offset const,
      stop_id const dep_s_id, time const prev_arrival,
      time const stop_departure) {
    if (!checked_tt) {
      auto const transfer_time = tt.transfer_times_[dep_s_id];
      return prev_arrival + transfer_time <= stop_departure;
    } else {
      return true;
    }
  }

  template <typename Data>
  _mark_cuda_rel_ inline static void set_departure_stop(Data&,
                                                        stop_offset const) {}

  template <typename Data>
  inline static bool is_rescan_from_stop_needed(uint32_t, Data const&,
                                                uint32_t) {
    return false;
  }

  template <typename Data, typename Timetable>
  _mark_cuda_rel_ inline static void update_aggregate(Data&, Timetable const&,
                                                      stop_offset const,
                                                      stop_times_index const) {}

  template <typename Data>
  _mark_cuda_rel_ inline static void reset_aggregate(trait_id const, Data&,
                                                     trait_id const) {}

#if defined(MOTIS_CUDA)
  template <typename Data>
  __device__ inline static void propagate_and_merge_if_needed(unsigned const,
                                                              Data&, bool const,
                                                              bool const) {}

  template <typename Data, typename Timetable>
  __device__ inline static void calculate_aggregate(Data&, Timetable const&,
                                                    time const* const,
                                                    stop_times_index const,
                                                    stop_times_index const) {}

  template <typename Data>
  __device__ inline static void carry_to_next_stage(unsigned const, Data&) {}
#endif

  template <typename Data>
  inline static std::vector<trait_id> get_feasible_trait_ids(trait_id const,
                                                             trait_id const,
                                                             Data const&) {
    return std::vector<trait_id>{0};
  }

  // giving the neutral element of the conjunction
  inline static bool dominates(trait_id const, trait_id const&,
                               trait_id const&) {
    return true;
  }

  inline static void fill_journey(trait_id const, journey&, trait_id const) {}

  template <typename Data>
  _mark_cuda_rel_ inline static void fill_aggregate(trait_id const, Data&,
                                                    trait_id const) {}
};

}  // namespace motis::raptor
