#pragma once

#include "motis/core/journey/journey.h"
#include "motis/protocol/ConnectionStatus_generated.h"

namespace motis {

inline ConnectionStatus status_to_fbs(journey::connection_status const s) {
  switch (s) {
    case journey::connection_status::OK: return ConnectionStatus_OK;
    case journey::connection_status::INTERCHANGE_INVALID:
      return ConnectionStatus_INTERCHANGE_INVALID;
    case journey::connection_status::TRAIN_HAS_BEEN_CANCELED:
      return ConnectionStatus_TRAIN_HAS_BEEN_CANCELED;
    case journey::connection_status::INVALID: return ConnectionStatus_INVALID;
    default: return ConnectionStatus_OK;
  }
}

inline journey::connection_status status_from_fbs(ConnectionStatus const s) {
  switch (s) {
    case ConnectionStatus_OK: return journey::connection_status::OK;
    case ConnectionStatus_INTERCHANGE_INVALID:
      return journey::connection_status::INTERCHANGE_INVALID;
    case ConnectionStatus_TRAIN_HAS_BEEN_CANCELED:
      return journey::connection_status::TRAIN_HAS_BEEN_CANCELED;
    case ConnectionStatus_INVALID: return journey::connection_status::INVALID;
    default: return journey::connection_status::OK;
  }
}

}  // namespace motis
