#pragma once

#include "motis/raptor/criteria/criteria_config.h"
#include "motis/raptor/criteria/criteria_helper.h"
#include "motis/raptor/criteria/traits.h"
#include "motis/raptor/criteria/traits/max_occupancy.h"
#include "motis/raptor/criteria/traits/min_transfer_times.h"
#include "motis/raptor/criteria/traits/time_slotted_occupancy.h"

namespace motis::raptor {

using Default = criteria_config<traits<>>;
using MaxOccupancy = criteria_config<traits<trait_max_occupancy>>;
using TimeSlottedOccupancy = criteria_config<traits<trait_time_slotted_occupancy>>;
using MinTransferTimes = criteria_config<traits<trait_min_transfer_times>>;


#define RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(DO, ACCESSOR) \
  DO(MaxOccupancy, ACCESSOR)                             \
  DO(TimeSlottedOccupancy, ACCESSOR)                     \
  DO(MinTransferTimes, ACCESSOR)

enum class raptor_criteria_config {
  Default,
  RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_ENUM_VALUE, )
};

inline raptor_criteria_config get_criteria_config_from_search_type(
    routing::SearchType const st) {
  switch (st) {
    case routing::SearchType_Default:
      return raptor_criteria_config::Default;
      RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(CASE_SEARCH_TYPE_TO_ENUM,
                                         raptor_criteria_config)
    default: throw std::system_error{access::error::not_implemented};
  }
}

inline uint32_t get_trait_size_for_criteria_config(
    raptor_criteria_config const rc) {
  switch (rc) {
    case raptor_criteria_config::Default:
      return Default::trait_size();  //^= 1

      RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(CASE_TRAIT_SIZE_FOR_CRITERIA_CONFIG,
                                         raptor_criteria_config)
    default: throw std::system_error{access::error::not_implemented};
  }
}

inline std::string get_string_for_criteria_config(
    raptor_criteria_config const config) {
  switch (config) {
    case raptor_criteria_config::Default:
      return "Default";

      RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(CASE_ENUM_TO_STRING,
                                         raptor_criteria_config)

    default: throw std::system_error{access::error::not_implemented};
  }
}

}  // namespace motis::raptor
