import { sendRequest } from "./api";
import { TripId } from "./base";

export function sendPaxMonTripLoadInfoRequest(trip: TripId): Promise<Response> {
  return sendRequest("/paxmon/trip_load_info", "TripId", trip);
}
