#pragma once

#include <tuple>

#include "motis/raptor/raptor_util.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

template <typename... TraitData>
struct trait_data : public TraitData... {};

template <typename... Trait>
struct traits;

template <typename FirstTrait, typename... RestTraits>
struct traits<FirstTrait, RestTraits...> {
  using TraitsData = trait_data<FirstTrait, RestTraits...>;

  __mark_cuda_rel__ inline static uint32_t size() {
    auto size = FirstTrait::value_range_size();
    return size * traits<RestTraits...>::size();
  }

  inline static void get_trait_data(uint32_t const total_size, TraitsData& dt,
                                    uint32_t const trait_offset) {
    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, trait_offset);

    FirstTrait::fill_trait_data_from_idx(dt, first_trait_idx);

    traits<RestTraits...>::get_trait_data(rest_trait_size, dt,
                                          rest_trait_offset);
  }

  template <typename Timetable>
  inline static bool trip_matches_traits(TraitsData const& dt,
                                         Timetable const& tt,
                                         uint32_t const r_id,
                                         uint32_t const t_id,
                                         uint32_t const dep_offset,
                                         uint32_t const arr_offset) {
    return FirstTrait::trip_matches_trait(dt, tt, r_id, t_id, dep_offset,
                                          arr_offset) &&
           traits<RestTraits...>::trip_matches_traits(dt, tt, r_id, t_id,
                                                      dep_offset, arr_offset);
  }

  inline static bool is_update_required(uint32_t total_size,
                                        TraitsData const& td,
                                        uint32_t t_offset) {

    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_update_required(td, first_trait_idx) &&
           traits<RestTraits...>::is_update_required(rest_trait_offset, td,
                                                     rest_trait_offset);
  }

  inline static bool is_trait_satisfied(uint32_t total_size,
                                        TraitsData const& td,
                                        uint32_t t_offset) {
    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_trait_satisfied(td, first_trait_idx) &&
           traits<RestTraits...>::is_trait_satisfied(rest_trait_size, td,
                                                     rest_trait_offset);
  }

  inline static bool is_rescan_from_stop_needed(uint32_t total_size,
                                                TraitsData const& td,
                                                uint32_t t_offset) {
    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_rescan_from_stop_needed(td, first_trait_idx) ||
           traits<RestTraits...>::is_rescan_from_stop_needed(
               rest_trait_size, td, rest_trait_offset);
  }

  // helper to aggregate values while progressing through the route stop by stop
  template <typename Timetable>
  inline static void update_aggregate(TraitsData& aggregate_dt,
                                      Timetable const& tt, uint32_t const r_id,
                                      uint32_t const t_id,
                                      uint32_t const s_offset,
                                      uint32_t const sti) {
    FirstTrait::update_aggregate(aggregate_dt, tt, r_id, t_id, s_offset, sti);
    traits<RestTraits...>::update_aggregate(aggregate_dt, tt, r_id, t_id,
                                            s_offset, sti);
  }

  // reset the aggregate everytime the departure station changes
  inline static void reset_aggregate(TraitsData& aggregate_dt) {
    FirstTrait::reset_aggregate(aggregate_dt);
    traits<RestTraits...>::reset_aggregate(aggregate_dt);
  }

  inline static bool dominates(TraitsData const& to_dominate,
                               TraitsData const& dominating) {
    return FirstTrait::dominates(to_dominate, dominating) &&
           traits<RestTraits...>::dominates(to_dominate, dominating);
  }

  inline static std::tuple<uint32_t, uint32_t, uint32_t> _trait_values(
      uint32_t const total_size, uint32_t const t_offset) {
    auto const first_value_size = FirstTrait::value_range_size();
    auto const rest_trait_size = total_size / first_value_size;

    auto const first_trait_idx = t_offset / rest_trait_size;
    auto const rest_trait_offset = t_offset % rest_trait_size;

    return std::make_tuple(rest_trait_size, first_trait_idx, rest_trait_offset);
  }

private:
};

template <>
struct traits<> {
  using TraitsData = trait_data<>;

  __mark_cuda_rel__ inline static uint32_t size() { return 1; }

  template <typename Data>
  inline static void get_trait_data(uint32_t const _1, Data& _2,
                                    uint32_t const _3) {}

  template <typename Data, typename Timetable>
  inline static bool trip_matches_traits(Data const& dt, Timetable const& tt,
                                         uint32_t const r_id,
                                         uint32_t const t_id,
                                         uint32_t const dep_offset,
                                         uint32_t const arr_offset) {
    return true;
  }

  template <typename Data>
  inline static bool is_update_required(uint32_t _1, Data const& _2,
                                        uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template <typename Data>
  inline static bool is_trait_satisfied(uint32_t _1, Data const& _2,
                                        uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template<typename Data>
  inline static bool is_rescan_from_stop_needed(uint32_t _1, Data const& _2,
                                       uint32_t _3) {
    return false;
  }

  template <typename Data, typename Timetable>
  inline static void update_aggregate(Data& _1, Timetable const& _2,
                                      uint32_t const _3, uint32_t const _4,
                                      uint32_t const _5, uint32_t const _6) {}

  template <typename Data>
  inline static void reset_aggregate(Data& _1) {}

  // giving the neutral element of the conjunction
  template <typename Data>
  inline static bool dominates(Data const& _1, Data const& _2) {
    return true;
  }
};

}  // namespace motis::raptor