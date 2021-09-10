import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { TripId } from "./protocol/motis";
import {
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonGetGroupsInTripResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonStatusResponse,
} from "./protocol/motis/paxmon";
import { useQuery, UseQueryResult } from "react-query";

export async function sendPaxMonStatusRequest(): Promise<PaxMonStatusResponse> {
  const msg = await sendRequest("/paxmon/status");
  verifyContentType(msg, "PaxMonStatusResponse");
  return msg.content as PaxMonStatusResponse;
}

export function usePaxMonStatusQuery(): UseQueryResult<PaxMonStatusResponse> {
  return useQuery(queryKeys.status(), sendPaxMonStatusRequest, {
    refetchInterval: 30 * 1000,
    refetchOnWindowFocus: true,
    staleTime: 0,
    notifyOnChangeProps: "tracked",
  });
}

export async function sendPaxMonTripLoadInfosRequest(
  content: PaxMonGetTripLoadInfosRequest
): Promise<PaxMonGetTripLoadInfosResponse> {
  const msg = await sendRequest(
    "/paxmon/trip_load_info",
    "PaxMonGetTripLoadInfosRequest",
    content
  );
  verifyContentType(msg, "PaxMonGetTripLoadInfosResponse");
  return msg.content as PaxMonGetTripLoadInfosResponse;
}

export async function sendPaxMonFindTripsRequest(
  content: PaxMonFindTripsRequest
): Promise<PaxMonFindTripsResponse> {
  const msg = await sendRequest(
    "/paxmon/find_trips",
    "PaxMonFindTripsRequest",
    content
  );
  verifyContentType(msg, "PaxMonFindTripsResponse");
  return msg.content as PaxMonFindTripsResponse;
}

export function usePaxMonFindTripsQuery(
  trainNr?: number
): UseQueryResult<PaxMonFindTripsResponse> {
  return useQuery(
    queryKeys.findTrips(trainNr),
    () =>
      sendPaxMonFindTripsRequest({
        universe: 0,
        train_nr: trainNr || 0,
        only_trips_with_paxmon_data: true,
        filter_class: false,
        max_class: 0,
      }),
    { enabled: trainNr != undefined && !isNaN(trainNr) }
  );
}

export async function sendPaxMonGroupsInTripRequest(
  trip: TripId
): Promise<PaxMonGetGroupsInTripResponse> {
  const msg = await sendRequest(
    "/paxmon/groups_in_trip",
    "PaxMonGetGroupsInTripRequest",
    {
      trip,
      include_grouped_by_destination: true,
    }
  );
  verifyContentType(msg, "PaxMonGetGroupsInTripResponse");
  return msg.content as PaxMonGetGroupsInTripResponse;
}

export function usePaxMonGroupsInTripQuery(
  tripId: TripId
): UseQueryResult<PaxMonGetGroupsInTripResponse> {
  return useQuery(queryKeys.tripGroups(tripId), () =>
    sendPaxMonGroupsInTripRequest(tripId)
  );
}

export const queryKeys = {
  all: ["paxmon"] as const,
  status: () => [...queryKeys.all, "status"] as const,
  findTrips: (trainNr?: number) =>
    [...queryKeys.all, "findTrips", trainNr] as const,
  tripLoad: (tripId: TripId) =>
    [...queryKeys.all, "trip", "load", { tripId }] as const,
  tripGroups: (tripId: TripId) =>
    [...queryKeys.all, "trip", "groups", { tripId }] as const,
};
