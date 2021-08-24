import { sendRequest } from "./api";
import { TripId } from "./base";

export function sendPaxMonTripLoadInfoRequest(trip: TripId): Promise<Response> {
  return sendRequest({
    destination: { target: "/paxmon/trip_load_info" },
    content_type: "TripId",
    content: trip,
  });
}
