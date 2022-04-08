#pragma once

#include <tuple>
#include <type_traits>

#include "motis/raptor/raptor_util.h"
#include "motis/raptor/types.h"

namespace motis {
struct journey;
}

namespace motis::raptor {
struct raptor_route;
struct cpu_mark_store;

enum class CalcMethod { Shfl, Flat };

template <typename... TraitData>
struct criteria_data : public TraitData... {
  static_assert(
      std::conjunction<std::is_default_constructible<TraitData>...>::value,
      "All trait data structs need to be default constructible!");

  _mark_cuda_rel_ criteria_data(raptor_route const* route,
                                trait_id const total_size,
                                trait_id const initial_t_offset)
      : route_{route},
        total_size_{total_size},
        initial_t_offset_{initial_t_offset} {}

  raptor_route const* route_;
  trip_id trip_id_{invalid<trip_id>};

  trait_id total_size_;
  trait_id initial_t_offset_;
};

struct default_transfer_time_calculator {

  template <typename TraitsData, typename Timetable>
  _mark_cuda_rel_ inline static time get_transfer_time(TraitsData const&,
                                                       Timetable const& tt,
                                                       stop_id const s_id) {
    if (!valid(s_id)) return invalid<time>;

    return tt.transfer_times_[s_id];
  }
};

template <typename Data, typename Traits, typename TransferTimeCalculator,
          CalcMethod calc>
struct criteria_config : Data {

  constexpr static auto USES_SHFL_CALC = calc == CalcMethod::Shfl;

  /***
   * Defines how many arrival values per stop will be allocated and
   * possibly written to.
   */
  constexpr static trait_id TRAITS_SIZE = Traits::SIZE;

  /***
   * Defines whether this configuration requires values to be
   * distributed along the dimensions
   */
  constexpr static bool REQ_DIMENSION_PROPAGATION =
      Traits::REQ_DIMENSION_PROPAGATION;

  constexpr static trait_id SWEEP_BLOCK_SIZE = Traits::SWEEP_BLOCK_SIZE;

  _mark_cuda_rel_ criteria_config(raptor_route const* route,
                                  trait_id const initial_offset)
      : Data(route, TRAITS_SIZE, initial_offset) {}

  _mark_cuda_rel_ inline static arrival_id get_arrival_idx(
      stop_id const stop_idx, trait_id const trait_offset = 0) {
    return stop_idx * TRAITS_SIZE + trait_offset;
  }

  _mark_cuda_rel_ inline trait_id get_write_to_trait_id() {
    return Traits::get_write_to_trait_id(*this);
  }

  template <typename Timetable>
  _mark_cuda_rel_ inline bool check_and_set_departure_stop(
      Timetable const& tt, stop_offset const dep_offset,
      stop_id const dep_stop_id, time const prev_arrival,
      time const stop_departure) {
    auto const feasible =
        Traits::check_departure_stop(false, tt, *this, dep_offset, dep_stop_id,
                                     prev_arrival, stop_departure);

    if (feasible) Traits::set_departure_stop(*this, dep_offset);

    return feasible;
  }

  template <typename Timetable>
  _mark_cuda_rel_ inline static time get_transfer_time(Timetable const& tt,
                                                       trait_id const t_offset,
                                                       stop_id const s_id) {
    criteria_config<Data, Traits, TransferTimeCalculator, calc> aggregate{
        nullptr, t_offset};
    Traits::fill_aggregate(TRAITS_SIZE, aggregate, t_offset);
    return TransferTimeCalculator::get_transfer_time(aggregate, tt, s_id);
  }

  _mark_cuda_rel_ inline bool is_satisfied(trait_id t_offset) {
    return Traits::is_trait_satisfied(TRAITS_SIZE, *this, t_offset);
  }

  //****************************************************************************
  // Only used by CPU RAPTOR
  inline bool is_rescan_from_stop_needed(trait_id t_offset) {
    return Traits::is_rescan_from_stop_needed(TRAITS_SIZE, *this, t_offset);
  }
  //****************************************************************************

  template <typename Timetable>
  _mark_cuda_rel_ inline void update_from_stop(
      Timetable const& tt, stop_offset const s_offset,
      stop_times_index const current_sti) {
    Traits::update_aggregate(*this, tt, s_offset, current_sti);
  }

  _mark_cuda_rel_ inline void reset(trip_id const t_id,
                                    trait_id const initial_offset) {
    this->trip_id_ = t_id;
    Traits::reset_aggregate(TRAITS_SIZE, *this, initial_offset);
  }

#if defined(MOTIS_CUDA)
  __device__ inline void propagate_along_warp(unsigned const mask,
                                              bool const is_departure_stop,
                                              bool const write_value) {
    Traits::propagate_and_merge_if_needed(mask, *this, is_departure_stop,
                                          write_value);
  }

  __device__ inline void carry_to_next_stage(unsigned const mask) {
    Traits::carry_to_next_stage(mask, *this);
  }

  template <typename Timetable>
  __device__ inline void calculate(Timetable const& tt,
                                   time const* const prev_arrivals,
                                   stop_times_index const dep_sti,
                                   stop_times_index const arr_sti) {
    Traits::calculate_aggregate(*this, tt, prev_arrivals, dep_sti, arr_sti);
  }

  _mark_cuda_rel_ static inline void perform_stop_arrival_sweeping_gpu(
      stop_id const s_id, time* const arrivals,
      uint32_t* station_marks) {
    Traits::perform_stop_arrival_sweeping_gpu(TRAITS_SIZE, s_id, arrivals, station_marks);
  }
#endif

  static inline void perform_stop_arrival_sweeping_cpu(
      stop_id const s_id, time* const arrivals,
      cpu_mark_store& station_marks) {
    Traits::perform_stop_arrival_sweeping_cpu(TRAITS_SIZE, s_id, arrivals, station_marks);
  }

  inline std::vector<trait_id> get_feasible_traits(
      trait_id const initial_offset) {
    auto feasible =
        Traits::get_feasible_trait_ids(TRAITS_SIZE, initial_offset, *this);
    std::reverse(std::begin(feasible), std::end(feasible));
    return feasible;
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  inline static bool dominates(trait_id const to_dominate,
                               trait_id const dominating) {
    return Traits::dominates(TRAITS_SIZE, to_dominate, dominating);
  }

  inline static void fill_journey(journey& journey, trait_id const t_offset) {
    Traits::fill_journey(TRAITS_SIZE, journey, t_offset);
  }
};

}  // namespace motis::raptor
