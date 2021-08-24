import { sendRequest } from "./api";
import { PaxMonFindTripsRequest } from "./paxmon";

export function sendPaxMonFindTripsRequest(
  options: PaxMonFindTripsRequest
): Promise<Response> {
  return sendRequest({
    destination: { target: "/paxmon/find_trips" },
    content_type: "PaxMonFindTripsRequest",
    content: {
      universe: 0,
      only_trips_with_paxmon_data: true,
      filter_class: false,
      max_class: 0,
      ...options,
    },
  });
}
