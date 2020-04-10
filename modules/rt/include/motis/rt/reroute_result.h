#pragma once

namespace motis::rt {

enum class reroute_result {
  OK,
  TRIP_NOT_FOUND,
  EVENT_COUNT_MISMATCH,
  STATION_MISMATCH,
  EVENT_ORDER_MISMATCH,
  RULE_SERVICE_REROUTE_NOT_SUPPORTED
};

}  // namespace motis::rt
