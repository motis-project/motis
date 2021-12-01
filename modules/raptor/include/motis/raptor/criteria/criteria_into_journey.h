#pragma once

#include "motis/core/journey/journey.h"
#include "motis/raptor/criteria/configs.h"

namespace motis::raptor {

template <typename CriteriaData>
inline void fill_criteria_into_journey(journey& j, CriteriaData const& data) {}

template <>
inline void fill_criteria_into_journey(journey& j,
                                       Default::CriteriaData const& data) {}

template <>
inline void fill_criteria_into_journey(journey& j,
                                       MaxOccupancy::CriteriaData const& data) {
  j.max_occupancy_ = data.max_occupancy_;
}

template <>
inline void fill_criteria_into_journey(
    journey& j, MinTransferTimes::CriteriaData const& data) {
  j.min_transfer_time_ = data.min_transfer_time_idx_ * 5;
}

template <>
inline void fill_criteria_into_journey(
    journey& j, TimeSlottedOccupancy::CriteriaData const& data) {
  j.time_slotted_occupancy_ = data.summed_occ_time_;
}

}  // namespace motis::raptor