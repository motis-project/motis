import { sendRequest } from "./api";
import { TripId } from "./base";

export function sendPaxMonGroupsInTripRequest(trip: TripId): Promise<Response> {
  return sendRequest({
    destination: { target: "/paxmon/groups_in_trip" },
    content_type: "PaxMonGetGroupsInTripRequest",
    content: { trip, include_grouped_by_destination: true },
  });
}
