#pragma once

#include <tuple>

#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"

namespace motis::raptor {

template <typename... TraitData>
struct raptor_data : public TraitData... {
  route_id route_id_;
  trip_id trip_id_;

  trait_id departure_trait_id_;
};

template <typename... Trait>
struct traits;

template <typename FirstTrait, typename... RestTraits>
struct traits<FirstTrait, RestTraits...> {
  using TraitsData = raptor_data<FirstTrait, RestTraits...>;

  __mark_cuda_rel__ inline static trait_id size() {
    auto size = FirstTrait::index_range_size();
    return size * traits<RestTraits...>::size();
  }

  // TODO get rid of
  __mark_cuda_rel__ inline static bool is_update_required(trait_id total_size,
                                                          TraitsData const& td,
                                                          trait_id t_offset) {

    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_update_required(td, first_dimension_idx) &&
           traits<RestTraits...>::is_update_required(rest_trait_offset, td,
                                                     rest_trait_offset);
  }

  __mark_cuda_rel__ inline static trait_id get_write_to_trait_id(
      TraitsData const& d) {
    auto const first_dimension_idx = FirstTrait::get_write_to_dimension_id(d);
    auto const first_dim_step_width = size();

    auto const rest_trait_offset =
        traits<RestTraits...>::get_write_to_trait_id(d);

    return (first_dim_step_width * first_dimension_idx) + rest_trait_offset;
  }

  inline static bool is_trait_satisfied(trait_id total_size,
                                        TraitsData const& td,
                                        trait_id t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_trait_satisfied(td, first_dimension_idx) &&
           traits<RestTraits...>::is_trait_satisfied(rest_trait_size, td,
                                                     rest_trait_offset);
  }

  //****************************************************************************
  // Only used by CPU RAPTOR
  inline static bool is_rescan_from_stop_needed(trait_id total_size,
                                                TraitsData const& td,
                                                trait_id t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_rescan_from_stop_needed(td, first_dimension_idx) ||
           traits<RestTraits...>::is_rescan_from_stop_needed(
               rest_trait_size, td, rest_trait_offset);
  }
  //****************************************************************************

  // helper to aggregate values while progressing through the route stop by stop
  template <typename Timetable>
  __mark_cuda_rel__ inline static void update_aggregate(
      TraitsData& aggregate_dt, Timetable const& tt,
      time const* const previous_arrivals, stop_offset const s_offset,
      stop_times_index const current_sti, trait_id const total_trait_size) {

    FirstTrait::update_aggregate(aggregate_dt, tt, previous_arrivals, s_offset,
                                 current_sti, total_trait_size);

    traits<RestTraits...>::update_aggregate(aggregate_dt, tt, previous_arrivals,
                                            s_offset, current_sti,
                                            total_trait_size);
  }

  // reset the aggregate everytime the departure station changes

  __mark_cuda_rel__ inline static void reset_aggregate(
      trait_id const total_size, TraitsData& aggregate_dt,
      trait_id const initial_t_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, initial_t_offset);

    FirstTrait::reset_aggregate(aggregate_dt, first_dimension_idx);
    traits<RestTraits...>::reset_aggregate(rest_trait_size, aggregate_dt,
                                           rest_trait_offset);
  }

#if defined(MOTIS_CUDA)
  __device__ inline static void propagate_and_merge_if_needed(
      unsigned const mask, TraitsData& aggregate, bool const predicate) {
    FirstTrait::propagate_and_merge_if_needed(aggregate, mask, predicate);
    traits<RestTraits...>::propagate_and_merge_if_needed(mask, aggregate,
                                                         predicate);
  }

  __device__ inline static void carry_to_next_stage(unsigned const mask,
                                                    TraitsData& aggregate) {
    FirstTrait::carry_to_next_stage(mask, aggregate);
    traits<RestTraits...>::carry_to_next_stage(mask, aggregate);
  }

#endif

  inline static void get_trait_data(trait_id const total_size, TraitsData& dt,
                                    trait_id const trait_offset) {
    auto const [rest_trait_size, first_dimension_idx, rest_trait_offset] =
        _trait_values(total_size, trait_offset);

    FirstTrait::fill_trait_data_from_idx(dt, first_dimension_idx);

    traits<RestTraits...>::get_trait_data(rest_trait_size, dt,
                                          rest_trait_offset);
  }

  inline static bool dominates(TraitsData const& to_dominate,
                               TraitsData const& dominating) {
    return FirstTrait::dominates(to_dominate, dominating) &&
           traits<RestTraits...>::dominates(to_dominate, dominating);
  }

  __mark_cuda_rel__ inline static std::tuple<trait_id, dimension_id, trait_id>
  _trait_values(trait_id const total_size, trait_id const t_offset) {
    auto const first_value_size = FirstTrait::index_range_size();
    auto const rest_trait_size = total_size / first_value_size;

    auto const first_dimension_idx = t_offset / rest_trait_size;
    auto const rest_trait_offset = t_offset % rest_trait_size;

    return std::make_tuple(rest_trait_size, first_dimension_idx,
                           rest_trait_offset);
  }

private:
};

template <>
struct traits<> {
  using TraitsData = raptor_data<>;

  __mark_cuda_rel__ inline static uint32_t size() { return 1; }

  template <typename Data>
  inline static void get_trait_data(uint32_t const _1, Data& _2,
                                    uint32_t const _3) {}

  template<typename Data>
  __mark_cuda_rel__ inline static trait_id get_write_to_trait_id(
      Data const& d){}

  template <typename Data>
  __mark_cuda_rel__ inline static bool is_update_required(uint32_t _1,
                                                          Data const& _2,
                                                          uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template <typename Data>
  inline static bool is_trait_satisfied(uint32_t _1, Data const& _2,
                                        uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template <typename Data>
  inline static bool is_rescan_from_stop_needed(uint32_t _1, Data const& _2,
                                                uint32_t _3) {
    return false;
  }

  template <typename Data, typename Timetable>
  __mark_cuda_rel__ inline static void update_aggregate(
      Data& aggregate_dt, Timetable const& tt,
      time const* const previous_arrivals, stop_offset const s_offset,
      stop_times_index const current_sti, trait_id const total_trait_size) {}

  template <typename Data>
  __mark_cuda_rel__ inline static void reset_aggregate(
      trait_id const total_size, Data& aggregate_dt,
      trait_id const initial_t_offset) {}

#if defined(MOTIS_CUDA)
  template <typename Data>
  __device__ inline static void propagate_and_merge_if_needed(unsigned const _1,
                                                              Data& _2,
                                                              bool const _3) {}

  template <typename Data>
  __device__ inline static void carry_to_next_stage(unsigned const _1,
                                                    Data& _2) {}
#endif

  // giving the neutral element of the conjunction
  template <typename Data>
  inline static bool dominates(Data const& _1, Data const& _2) {
    return true;
  }
};

}  // namespace motis::raptor