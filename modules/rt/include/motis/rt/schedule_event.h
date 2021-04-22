#pragma once

#include <cstdint>
#include <tuple>

#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

namespace motis::rt {

struct schedule_event {
  schedule_event(primary_trip_id trp_id, uint32_t station_idx,
                 motis::time schedule_time, event_type ev_type)
      : trp_id_(trp_id),
        station_idx_(station_idx),
        schedule_time_(schedule_time),
        ev_type_(ev_type) {}

  friend bool operator<(schedule_event const& a, schedule_event const& b) {
    return std::tie(a.trp_id_, a.station_idx_, a.schedule_time_, a.ev_type_) <
           std::tie(b.trp_id_, b.station_idx_, b.schedule_time_, b.ev_type_);
  }

  friend bool operator==(schedule_event const& a, schedule_event const& b) {
    return std::tie(a.trp_id_, a.station_idx_, a.schedule_time_, a.ev_type_) ==
           std::tie(b.trp_id_, b.station_idx_, b.schedule_time_, b.ev_type_);
  }

  primary_trip_id trp_id_;
  uint32_t station_idx_;
  motis::time schedule_time_;
  event_type ev_type_;
};

}  // namespace motis::rt
