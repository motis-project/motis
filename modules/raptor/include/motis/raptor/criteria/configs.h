#pragma once

#include "motis/raptor/criteria/criteria_config.h"
#include "motis/raptor/criteria/criteria_helper.h"
#include "motis/raptor/criteria/traits.h"
#include "motis/raptor/criteria/traits/max_occupancy.h"
#include "motis/raptor/criteria/traits/time_slotted_occupancy.h"
#include "motis/raptor/criteria/traits/tranfer_classes.h"

namespace motis::raptor {

using Default = criteria_config<traits<>, CalcMethod::Flat>;

using MaxOccupancy =
    criteria_config<traits<trait_max_occupancy>, CalcMethod::Flat>;

using MaxOccupancyShfl =
    criteria_config<traits<trait_max_occupancy>, CalcMethod::Shfl>;

// using TimeSlottedOccupancy =
//     criteria_config<traits<trait_time_slotted_occupancy<64>>,
//     CalcMethod::Flat>;

// using TimeSlottedOccupancyShfl =
//     criteria_config<traits<trait_time_slotted_occupancy<64>>,
//     CalcMethod::Shfl>;

using Tso96 =
    criteria_config<traits<trait_time_slotted_occupancy<96>>, CalcMethod::Flat>;
using Tso90 =
    criteria_config<traits<trait_time_slotted_occupancy<90>>, CalcMethod::Flat>;
using Tso80 =
    criteria_config<traits<trait_time_slotted_occupancy<80>>, CalcMethod::Flat>;
using Tso72 =
    criteria_config<traits<trait_time_slotted_occupancy<72>>, CalcMethod::Flat>;
using Tso64 =
    criteria_config<traits<trait_time_slotted_occupancy<64>>, CalcMethod::Flat>;
using Tso60 =
    criteria_config<traits<trait_time_slotted_occupancy<60>>, CalcMethod::Flat>;
using Tso48 =
    criteria_config<traits<trait_time_slotted_occupancy<48>>, CalcMethod::Flat>;
using Tso45 =
    criteria_config<traits<trait_time_slotted_occupancy<45>>, CalcMethod::Flat>;
using Tso40 =
    criteria_config<traits<trait_time_slotted_occupancy<40>>, CalcMethod::Flat>;
using Tso36 =
    criteria_config<traits<trait_time_slotted_occupancy<36>>, CalcMethod::Flat>;
using Tso32 =
    criteria_config<traits<trait_time_slotted_occupancy<32>>, CalcMethod::Flat>;
using Tso30 =
    criteria_config<traits<trait_time_slotted_occupancy<30>>, CalcMethod::Flat>;
using Tso24 =
    criteria_config<traits<trait_time_slotted_occupancy<24>>, CalcMethod::Flat>;
using Tso20 =
    criteria_config<traits<trait_time_slotted_occupancy<20>>, CalcMethod::Flat>;
using Tso18 =
    criteria_config<traits<trait_time_slotted_occupancy<18>>, CalcMethod::Flat>;
using Tso16 =
    criteria_config<traits<trait_time_slotted_occupancy<16>>, CalcMethod::Flat>;
using Tso12 =
    criteria_config<traits<trait_time_slotted_occupancy<12>>, CalcMethod::Flat>;
using Tso10 =
    criteria_config<traits<trait_time_slotted_occupancy<10>>, CalcMethod::Flat>;
using Tso08 =
    criteria_config<traits<trait_time_slotted_occupancy<8>>, CalcMethod::Flat>;
using Tso06 =
    criteria_config<traits<trait_time_slotted_occupancy<6>>, CalcMethod::Flat>;

using Tso96Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<96>>, CalcMethod::Shfl>;
using Tso90Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<90>>, CalcMethod::Shfl>;
using Tso80Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<80>>, CalcMethod::Shfl>;
using Tso72Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<72>>, CalcMethod::Shfl>;
using Tso64Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<64>>, CalcMethod::Shfl>;
using Tso60Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<60>>, CalcMethod::Shfl>;
using Tso48Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<48>>, CalcMethod::Shfl>;
using Tso45Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<45>>, CalcMethod::Shfl>;
using Tso40Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<40>>, CalcMethod::Shfl>;
using Tso36Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<36>>, CalcMethod::Shfl>;
using Tso32Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<32>>, CalcMethod::Shfl>;
using Tso30Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<30>>, CalcMethod::Shfl>;
using Tso24Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<24>>, CalcMethod::Shfl>;
using Tso20Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<20>>, CalcMethod::Shfl>;
using Tso18Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<18>>, CalcMethod::Shfl>;
using Tso16Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<16>>, CalcMethod::Shfl>;
using Tso12Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<12>>, CalcMethod::Shfl>;
using Tso10Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<10>>, CalcMethod::Shfl>;
using Tso08Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<8>>, CalcMethod::Shfl>;
using Tso06Shfl =
    criteria_config<traits<trait_time_slotted_occupancy<6>>, CalcMethod::Shfl>;

using MinTransferTimes =
    criteria_config<traits<trait_max_transfer_class>, CalcMethod::Flat>;

#define RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(DO, ACCESSOR) \
  DO(MaxOccupancy, ACCESSOR)                             \
  DO(MaxOccupancyShfl, ACCESSOR)                         \
  DO(MinTransferTimes, ACCESSOR)                         \
                                                         \
  DO(Tso96, ACCESSOR)                                    \
  DO(Tso90, ACCESSOR)                                    \
  DO(Tso80, ACCESSOR)                                    \
  DO(Tso72, ACCESSOR)                                    \
  DO(Tso64, ACCESSOR)                                    \
  DO(Tso60, ACCESSOR)                                    \
  DO(Tso48, ACCESSOR)                                    \
  DO(Tso45, ACCESSOR)                                    \
  DO(Tso40, ACCESSOR)                                    \
  DO(Tso36, ACCESSOR)                                    \
  DO(Tso32, ACCESSOR)                                    \
  DO(Tso30, ACCESSOR)                                    \
  DO(Tso24, ACCESSOR)                                    \
  DO(Tso20, ACCESSOR)                                    \
  DO(Tso18, ACCESSOR)                                    \
  DO(Tso16, ACCESSOR)                                    \
  DO(Tso12, ACCESSOR)                                    \
  DO(Tso10, ACCESSOR)                                    \
  DO(Tso08, ACCESSOR)                                    \
  DO(Tso06, ACCESSOR)                                    \
                                                         \
  DO(Tso96Shfl, ACCESSOR)                                \
  DO(Tso90Shfl, ACCESSOR)                                \
  DO(Tso80Shfl, ACCESSOR)                                \
  DO(Tso72Shfl, ACCESSOR)                                \
  DO(Tso64Shfl, ACCESSOR)                                \
  DO(Tso60Shfl, ACCESSOR)                                \
  DO(Tso48Shfl, ACCESSOR)                                \
  DO(Tso45Shfl, ACCESSOR)                                \
  DO(Tso40Shfl, ACCESSOR)                                \
  DO(Tso36Shfl, ACCESSOR)                                \
  DO(Tso32Shfl, ACCESSOR)                                \
  DO(Tso30Shfl, ACCESSOR)                                \
  DO(Tso24Shfl, ACCESSOR)                                \
  DO(Tso20Shfl, ACCESSOR)                                \
  DO(Tso18Shfl, ACCESSOR)                                \
  DO(Tso16Shfl, ACCESSOR)                                \
  DO(Tso12Shfl, ACCESSOR)                                \
  DO(Tso10Shfl, ACCESSOR)                                \
  DO(Tso08Shfl, ACCESSOR)                                \
  DO(Tso06Shfl, ACCESSOR)

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

inline trait_id get_trait_size_for_criteria_config(
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
