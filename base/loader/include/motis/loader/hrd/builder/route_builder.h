#pragma once

#include <cinttypes>
#include <map>
#include <vector>

#include "utl/parser/cstr.h"

#include "motis/schedule-format/Schedule_generated.h"

#include "motis/loader/hrd/builder/bitfield_builder.h"
#include "motis/loader/hrd/builder/provider_builder.h"
#include "motis/loader/hrd/builder/rule_service_builder.h"
#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/model/hrd_service.h"

namespace motis::loader::hrd {

struct stop_restrictions {
  friend bool operator<(stop_restrictions const& lhs,
                        stop_restrictions const& rhs) {
    if (rhs.eva_num_ < lhs.eva_num_) {
      return true;
    } else if (rhs.eva_num_ > lhs.eva_num_) {
      return false;
    }
    if (static_cast<int>(rhs.entering_allowed_) <
        static_cast<int>(lhs.entering_allowed_)) {
      return true;
    } else if (static_cast<int>(rhs.entering_allowed_) >
               static_cast<int>(lhs.entering_allowed_)) {
      return false;
    }
    if (static_cast<int>(rhs.leaving_allowed_) <
        static_cast<int>(lhs.leaving_allowed_)) {
      return true;
    } else if (static_cast<int>(rhs.entering_allowed_) >
               static_cast<int>(lhs.entering_allowed_)) {
      return false;
    }
    return false;
  }

  friend bool operator==(stop_restrictions const& lhs,
                         stop_restrictions const& rhs) {
    return lhs.eva_num_ == rhs.eva_num_ &&
           lhs.entering_allowed_ == rhs.entering_allowed_ &&
           lhs.leaving_allowed_ == rhs.leaving_allowed_;
  }

  int eva_num_;
  bool entering_allowed_;
  bool leaving_allowed_;
};

struct route_builder {
  flatbuffers64::Offset<Route> get_or_create_route(
      std::vector<hrd_service::stop> const&, station_builder&,
      flatbuffers64::FlatBufferBuilder&);

  std::map<std::vector<stop_restrictions>, flatbuffers64::Offset<Route>>
      routes_;
};

}  // namespace motis::loader::hrd
