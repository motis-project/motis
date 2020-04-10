#pragma once

#include "motis/core/schedule/event_type.h"
#include "motis/protocol/EventType_generated.h"

namespace motis {

inline event_type from_fbs(EventType const ev_type) {
  return ev_type == EventType::EventType_DEP ? event_type::DEP
                                             : event_type::ARR;
}

inline EventType to_fbs(event_type const ev_type) {
  return ev_type == event_type::DEP ? EventType_DEP : EventType_ARR;
}

}  // namespace motis
