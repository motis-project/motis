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

}  // namespace motis::raptor