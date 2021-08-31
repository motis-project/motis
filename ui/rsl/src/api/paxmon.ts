import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { Message, TripId } from "./protocol/motis";
import {
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonGetGroupsInTripResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonStatusResponse,
} from "./protocol/motis/paxmon";

export function sendPaxMonStatusRequest(): Promise<PaxMonStatusResponse> {
  return sendRequest("/paxmon/status").then((msg) => {
    verifyContentType(msg, "PaxMonStatusResponse");
    return msg.content as PaxMonStatusResponse;
  });
}

export function sendPaxMonInitForward(): Promise<Message> {
  return sendRequest("/paxmon/init_forward").then((msg) => {
    return verifyContentType(msg, "MotisNoMessage");
  });
}

export function sendPaxMonTripLoadInfosRequest(
  content: PaxMonGetTripLoadInfosRequest
): Promise<PaxMonGetTripLoadInfosResponse> {
  return sendRequest(
    "/paxmon/trip_load_info",
    "PaxMonGetTripLoadInfosRequest",
    content
  ).then((msg) => {
    verifyContentType(msg, "PaxMonGetTripLoadInfosResponse");
    return msg.content as PaxMonGetTripLoadInfosResponse;
  });
}

export function sendPaxMonFindTripsRequest(
  content: PaxMonFindTripsRequest
): Promise<PaxMonFindTripsResponse> {
  return sendRequest(
    "/paxmon/find_trips",
    "PaxMonFindTripsRequest",
    content
  ).then((msg) => {
    verifyContentType(msg, "PaxMonFindTripsResponse");
    return msg.content as PaxMonFindTripsResponse;
  });
}

export function sendPaxMonGroupsInTripRequest(
  trip: TripId
): Promise<PaxMonGetGroupsInTripResponse> {
  return sendRequest("/paxmon/groups_in_trip", "PaxMonGetGroupsInTripRequest", {
    trip,
    include_grouped_by_destination: true,
  }).then((msg) => {
    verifyContentType(msg, "PaxMonGetGroupsInTripResponse");
    return msg.content as PaxMonGetGroupsInTripResponse;
  });
}
