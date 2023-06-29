#pragma once

#include <string_view>

#include "boost/uuid/string_generator.hpp"

#include "flatbuffers/flatbuffers.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/station_access.h"
#include "motis/protocol/TripFormationMessage_generated.h"

namespace motis::rt {

inline std::string_view view(flatbuffers::String const* s) {
  return {s->c_str(), s->size()};
}

inline boost::uuids::uuid parse_uuid(std::string_view const sv) {
  return boost::uuids::string_generator{}(sv.begin(), sv.end());
}

inline bool get_primary_trip_id(schedule const& sched,
                                ris::HalfTripId const* hti,
                                primary_trip_id& ptid) {
  auto const tid = hti->id();
  auto const* st = find_station(sched, view(tid->station_id()));
  if (st != nullptr) {
    ptid.station_id_ = st->index_;
  }
  ptid.time_ = unix_to_motistime(sched.schedule_begin_, tid->time());
  ptid.train_nr_ = tid->train_nr();

  return st != nullptr && ptid.time_ != INVALID_TIME;
}

}  // namespace motis::rt
