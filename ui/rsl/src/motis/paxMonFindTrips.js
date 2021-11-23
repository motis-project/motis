import { sendRequest } from "./api";

export function sendPaxMonFindTripsRequest(options) {
  return sendRequest("/paxmon/find_trips", "PaxMonFindTripsRequest", {
    only_trips_with_paxmon_data: true,
    filter_class: false,
    max_class: 0,
    ...options,
  });
}
