import { sendRequest } from "./request";
import { PaxMonFindTripsRequest } from "./protocol/motis/paxmon";
import { TripId } from "./protocol/motis";

export function sendPaxMonStatusRequest(): Promise<Response> {
  return sendRequest("/paxmon/status");
}

export function sendPaxMonInitForward(): Promise<Response> {
  return sendRequest("/paxmon/init_forward");
}

export function sendPaxMonTripLoadInfoRequest(trip: TripId): Promise<Response> {
  return sendRequest("/paxmon/trip_load_info", "TripId", trip);
}

export function sendPaxMonFindTripsRequest(
  content: PaxMonFindTripsRequest
): Promise<Response> {
  return sendRequest("/paxmon/find_trips", "PaxMonFindTripsRequest", content);
}

export function sendPaxMonGroupsInTripRequest(trip: TripId): Promise<Response> {
  return sendRequest("/paxmon/groups_in_trip", "PaxMonGetGroupsInTripRequest", {
    trip,
    include_grouped_by_destination: true,
  });
}
