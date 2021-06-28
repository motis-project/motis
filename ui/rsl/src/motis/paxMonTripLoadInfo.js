import { sendRequest } from "./api";

export function sendPaxMonTripLoadInfoRequest(trip) {
  return sendRequest("/paxmon/trip_load_info", "TripId", trip);
}
