#pragma once

#include "motis/core/schedule/timestamp_reason.h"
#include "motis/protocol/TimestampReason_generated.h"

namespace motis {

inline TimestampReason to_fbs(timestamp_reason const r) {
  switch (r) {
    case timestamp_reason::REPAIR: return TimestampReason_REPAIR;
    case timestamp_reason::SCHEDULE: return TimestampReason_SCHEDULE;
    case timestamp_reason::IS: return TimestampReason_IS;
    case timestamp_reason::FORECAST: return TimestampReason_FORECAST;
    case timestamp_reason::PROPAGATION: return TimestampReason_PROPAGATION;
    default: return TimestampReason_SCHEDULE;
  }
}

inline timestamp_reason from_fbs(TimestampReason const r) {
  switch (r) {
    case TimestampReason_REPAIR: return timestamp_reason::REPAIR;
    case TimestampReason_SCHEDULE: return timestamp_reason::SCHEDULE;
    case TimestampReason_IS: return timestamp_reason::IS;
    case TimestampReason_FORECAST: return timestamp_reason::FORECAST;
    case TimestampReason_PROPAGATION: return timestamp_reason::PROPAGATION;
    default: return timestamp_reason::SCHEDULE;
  }
}

}  // namespace motis
