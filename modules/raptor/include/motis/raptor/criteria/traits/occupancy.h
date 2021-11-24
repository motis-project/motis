#pragma once

#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "cooperative_groups.h"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#endif

namespace motis::raptor {

constexpr uint8_t occ_value_range = 2;  // ^= 0-2
constexpr std::size_t occ_value_range_size = occ_value_range + 1;
constexpr std::size_t occ_total_value_range_size =
    max_raptor_round * occ_value_range_size;

struct trait_occupancy {
  // Trait Data
  uint8_t occupancy_;

  __mark_cuda_rel__ inline static std::size_t value_range_size() {
    return occ_total_value_range_size;
  }

  template <typename TraitsData>
  inline static void fill_trait_data_from_idx(TraitsData& dt,
                                              uint32_t const dim_idx) {
    // total occupancy linearly maps to the dimension idx
    dt.occupancy_ = dim_idx;
  }

  template <typename TraitsData>
  inline static bool is_rescan_from_stop_needed(TraitsData const& data,
                                                uint32_t trait_idx) {
    return trait_idx < data.occupancy_; //TODO
  }

  template<typename TraitsData>
  __mark_cuda_rel__ inline static bool is_update_required(TraitsData const& current_trip_data,
                                        uint32_t old_trip_idx) {
    return true;  // TODO
  }

  template<typename TraitsData>
  inline static bool is_trait_satisfied(TraitsData const& current_trip_data,
                                        uint32_t old_trip_idx) {
    return old_trip_idx == 0 && current_trip_data.occupancy_ == 0;  // TODO
  }

  template <typename TraitsData, typename Timetable>
  __mark_cuda_rel__ inline static void update_aggregate(TraitsData& aggregate_dt,
                                      Timetable const& tt, uint32_t const _1,
                                      uint32_t const _2, uint32_t const _3,
                                      uint32_t const sti) {
    auto const stop_occupancy = _read_occupancy(tt, _1, _2, _3, sti);
    aggregate_dt.occupancy_ =
        std::max(aggregate_dt.occupancy_, stop_occupancy);
  }

  template <typename Timetable>
  __mark_cuda_rel__ inline static uint8_t _read_occupancy(Timetable const& tt,
                                                          uint32_t const _1,
                                                          uint32_t const _2,
                                                          uint32_t const _3,
                                                          uint32_t const sti) {
    return tt.stop_attr_[sti].inbound_occupancy_;
  }

#if defined(MOTIS_CUDA)
  template <>
  __mark_cuda_rel__ inline static uint8_t _read_occupancy<device_gpu_timetable>(
      device_gpu_timetable const& tt, uint32_t const _1, uint32_t const _2,
      uint32_t const _3, uint32_t const sti) {
    return tt.stop_inb_occupancy_[sti];
  }

  template <typename TraitsData>
  __device__ inline static void propagate_and_merge_if_needed(
      unsigned const mask, TraitsData& aggregate, bool const predicate) {
    auto const prop_val = aggregate.occupancy_;
    auto const received = __shfl_up_sync(mask, prop_val, 1);
    if(predicate && aggregate.occupancy_ < received)
      aggregate.occupancy_ = received;
  }

  template<typename TraitsData>
  __device__ inline static void carry_to_next_stage(
      unsigned const mask, TraitsData& aggregate
      ) {
    auto const prop_val = aggregate.occupancy_;
    auto const received = __shfl_down_sync(mask, prop_val, 31);
    aggregate.occupancy_ = received;
  }

#endif

  template <typename TraitsData>
  __mark_cuda_rel__ inline static void reset_aggregate(TraitsData& aggregate_dt) {
    aggregate_dt.occupancy_ = 0;
  }

  template <typename TraitsData, typename Timetable>
  inline static bool trip_matches_trait(TraitsData const& dt,
                                        Timetable const& tt,
                                        uint32_t const r_id,
                                        uint32_t const t_id,
                                        uint32_t const dep_offset,
                                        uint32_t const arr_offset) {
    auto const route = tt.routes_[r_id];
    auto const dep_sti =
        route.index_to_stop_times_ + (route.stop_count_ * t_id) + dep_offset;
    auto const arr_sti =
        route.index_to_stop_times_ + (route.stop_count_ * t_id) + arr_offset;

    uint8_t occ = 0;
    for (auto sti = (dep_sti + 1); sti <= arr_sti; ++sti) {
      occ = std::max(occ, tt.stop_attr_[sti].inbound_occupancy_);
    }

    return occ <= dt.occupancy_;
  }

  // check if journey dominates candidate in max_occupancy
  template <typename TraitsData>
  static bool dominates(TraitsData const& to_dominate,
                        TraitsData const& dominating) {
    return dominating.occupancy_ <= to_dominate.occupancy_;
  }
};

}  // namespace motis::raptor