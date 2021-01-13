import { sendRequest } from "./api";

export function sendPaxMonStatusRequest(options) {
  return sendRequest("/paxmon/status", "PaxMonStatusRequest", {
    include_trips_affected_by_last_update: false,
    include_trips_with_critical_sections: false,
    ...options,
  });
}
