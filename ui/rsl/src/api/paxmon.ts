import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { MotisSuccess, TripId } from "./protocol/motis";
import {
  PaxMonDestroyUniverseRequest,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsResponse,
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonForkUniverseRequest,
  PaxMonForkUniverseResponse,
  PaxMonGetGroupsInTripRequest,
  PaxMonGetGroupsInTripResponse,
  PaxMonGetInterchangesRequest,
  PaxMonGetInterchangesResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonStatusRequest,
  PaxMonStatusResponse,
} from "./protocol/motis/paxmon";
import { useQuery, UseQueryResult } from "react-query";

export async function sendPaxMonStatusRequest(
  content: PaxMonStatusRequest
): Promise<PaxMonStatusResponse> {
  const msg = await sendRequest(
    "/paxmon/status",
    "PaxMonStatusRequest",
    content
  );
  verifyContentType(msg, "PaxMonStatusResponse");
  return msg.content as PaxMonStatusResponse;
}

export function usePaxMonStatusQuery(
  universe: number
): UseQueryResult<PaxMonStatusResponse> {
  return useQuery(
    queryKeys.status(universe),
    () => sendPaxMonStatusRequest({ universe }),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      staleTime: 0,
      notifyOnChangeProps: "tracked",
    }
  );
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
  universe: number,
  trainNr?: number
): UseQueryResult<PaxMonFindTripsResponse> {
  return useQuery(
    queryKeys.findTrips(universe, trainNr),
    () =>
      sendPaxMonFindTripsRequest({
        universe,
        train_nr: trainNr || 0,
        only_trips_with_paxmon_data: true,
        filter_class: false,
        max_class: 0,
      }),
    { enabled: trainNr != undefined && !isNaN(trainNr) }
  );
}

export async function sendPaxMonGroupsInTripRequest(
  content: PaxMonGetGroupsInTripRequest
): Promise<PaxMonGetGroupsInTripResponse> {
  const msg = await sendRequest(
    "/paxmon/groups_in_trip",
    "PaxMonGetGroupsInTripRequest",
    content
  );
  verifyContentType(msg, "PaxMonGetGroupsInTripResponse");
  return msg.content as PaxMonGetGroupsInTripResponse;
}

export function usePaxMonGroupsInTripQuery(
  content: PaxMonGetGroupsInTripRequest
): UseQueryResult<PaxMonGetGroupsInTripResponse> {
  return useQuery(queryKeys.tripGroups(content), () =>
    sendPaxMonGroupsInTripRequest(content)
  );
}

export async function sendPaxMonForkUniverseRequest(
  content: PaxMonForkUniverseRequest
): Promise<PaxMonForkUniverseResponse> {
  const msg = await sendRequest(
    "/paxmon/fork_universe",
    "PaxMonForkUniverseRequest",
    content
  );
  verifyContentType(msg, "PaxMonForkUniverseResponse");
  return msg.content as PaxMonForkUniverseResponse;
}

export async function sendPaxMonDestroyUniverseRequest(
  content: PaxMonDestroyUniverseRequest
): Promise<MotisSuccess> {
  const msg = await sendRequest(
    "/paxmon/destroy_universe",
    "PaxMonDestroyUniverseRequest",
    content
  );
  verifyContentType(msg, "MotisSuccess");
  return msg.content as MotisSuccess;
}

export async function sendPaxMonGetInterchangesRequest(
  content: PaxMonGetInterchangesRequest
): Promise<PaxMonGetInterchangesResponse> {
  const msg = await sendRequest(
    "/paxmon/get_interchanges",
    "PaxMonGetInterchangesRequest",
    content
  );
  verifyContentType(msg, "PaxMonGetInterchangesResponse");
  return msg.content as PaxMonGetInterchangesResponse;
}

export function usePaxMonGetInterchangesQuery(
  content: PaxMonGetInterchangesRequest
): UseQueryResult<PaxMonGetInterchangesResponse> {
  return useQuery(queryKeys.interchanges(content), () =>
    sendPaxMonGetInterchangesRequest(content)
  );
}

export async function sendPaxMonFilterTripsRequest(
  content: PaxMonFilterTripsRequest
): Promise<PaxMonFilterTripsResponse> {
  const msg = await sendRequest(
    "/paxmon/filter_trips",
    "PaxMonFilterTripsRequest",
    content
  );
  verifyContentType(msg, "PaxMonFilterTripsResponse");
  return msg.content as PaxMonFilterTripsResponse;
}

export function usePaxMonFilterTripsRequest(content: PaxMonFilterTripsRequest) {
  return useQuery(queryKeys.filterTrips(content), () =>
    sendPaxMonFilterTripsRequest(content)
  );
}

export const queryKeys = {
  all: ["paxmon"] as const,
  status: (universe: number) => [...queryKeys.all, "status", universe] as const,
  findTrips: (universe: number, trainNr?: number) =>
    [...queryKeys.all, "find_trips", universe, trainNr] as const,
  trip: () => [...queryKeys.all, "trip"] as const,
  tripLoad: (universe: number, tripId: TripId) =>
    [...queryKeys.trip(), "load", universe, { tripId }] as const,
  tripGroups: (req: PaxMonGetGroupsInTripRequest) =>
    [...queryKeys.trip(), "groups", req] as const,
  interchanges: (req: PaxMonGetInterchangesRequest) =>
    [...queryKeys.all, "interchanges", req] as const,
  filterTrips: (req: PaxMonFilterTripsRequest) =>
    [...queryKeys.all, "filter_trips", req] as const,
};
