#pragma once

#include <cstdint>
#include <algorithm>
#include <tuple>

namespace motis::raptor {

constexpr uint8_t max_occupancy = 2;

// linearly scale max_occupancy values to indices
constexpr uint32_t moc_value_range_size = max_occupancy + 1;

struct trait_max_occupancy {
  // Trait Data
  uint8_t max_occupancy_;

  inline static uint32_t value_range_size() { return moc_value_range_size; }

  template <typename TraitsData>
  inline static void fill_trait_data_from_idx(TraitsData& dt,
                                              uint32_t const dimension_idx) {
    // can be used as occupancy at idx 0
    //  maps to an occupancy value of 0
    dt.max_occupancy_ = dimension_idx;
  }

  template <typename TraitsData>
  inline static bool is_update_required(TraitsData const& data,
                                        uint32_t trait_idx) {
    return trait_idx >=
           data.max_occupancy_;  // for MOC there is trait_idx == trait_value
  }

  template <typename TraitsData>
  inline static bool is_trait_satisfied(TraitsData const& data,
                                        uint32_t trait_idx) {
    return trait_idx == 0 && data.max_occupancy_ == 0;
  }

  template <typename TraitsData, typename Timetable>
  inline static void update_aggregate(TraitsData& aggregate_dt,
                                      Timetable const& tt, uint32_t const _1,
                                      uint32_t const _2, uint32_t const _3,
                                      uint32_t const sti) {
    auto const stop_occupancy = tt.stop_attr_[sti].inbound_occupancy_;
    aggregate_dt.max_occupancy_ =
        std::max(aggregate_dt.max_occupancy_, stop_occupancy);
  }

  template <typename TraitsData>
  inline static void reset_aggregate(TraitsData& aggregate_dt) {
    aggregate_dt.max_occupancy_ = 0;
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

    uint8_t max_occ = 0;
    for (auto sti = (dep_sti + 1); sti <= arr_sti; ++sti) {
      max_occ = std::max(max_occ, tt.stop_attr_[sti].inbound_occupancy_);
    }

    return max_occ <= dt.max_occupancy_;
  }

  template <typename TraitsData>
  static bool dominates(TraitsData const& to_dominate,
                        TraitsData const& dominating) {
    return dominating.max_occupancy_ <= to_dominate.max_occupancy_;
  }
};

}  // namespace motis::raptor