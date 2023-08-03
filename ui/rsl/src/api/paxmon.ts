import {
  UseQueryOptions,
  UseQueryResult,
  useQuery,
} from "@tanstack/react-query";

import { verifyContentType } from "@/api/protocol/checks";
import { MotisSuccess, TripId } from "@/api/protocol/motis";
import {
  PaxMonCapacityStatusRequest,
  PaxMonCapacityStatusResponse,
  PaxMonDebugGraphRequest,
  PaxMonDebugGraphResponse,
  PaxMonDestroyUniverseRequest,
  PaxMonDetailedCapacityStatusRequest,
  PaxMonDetailedCapacityStatusResponse,
  PaxMonFilterGroupsRequest,
  PaxMonFilterGroupsResponse,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsResponse,
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonForkUniverseRequest,
  PaxMonForkUniverseResponse,
  PaxMonGetAddressableGroupsRequest,
  PaxMonGetAddressableGroupsResponse,
  PaxMonGetGroupsInTripRequest,
  PaxMonGetGroupsInTripResponse,
  PaxMonGetGroupsRequest,
  PaxMonGetGroupsResponse,
  PaxMonGetInterchangesRequest,
  PaxMonGetInterchangesResponse,
  PaxMonGetTripCapacityRequest,
  PaxMonGetTripCapacityResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonGetUniversesResponse,
  PaxMonGroupStatisticsRequest,
  PaxMonGroupStatisticsResponse,
  PaxMonKeepAliveRequest,
  PaxMonKeepAliveResponse,
  PaxMonMetricsRequest,
  PaxMonMetricsResponse,
  PaxMonStatusRequest,
  PaxMonStatusResponse,
} from "@/api/protocol/motis/paxmon";

import { sendRequest } from "@/api/request";

export async function sendPaxMonStatusRequest(
  content: PaxMonStatusRequest,
): Promise<PaxMonStatusResponse> {
  const msg = await sendRequest(
    "/paxmon/status",
    "PaxMonStatusRequest",
    content,
  );
  verifyContentType(msg, "PaxMonStatusResponse");
  return msg.content as PaxMonStatusResponse;
}

export function usePaxMonStatusQuery(
  universe: number,
): UseQueryResult<PaxMonStatusResponse> {
  return useQuery(
    queryKeys.status(universe),
    () => sendPaxMonStatusRequest({ universe }),
    {
      staleTime: 0,
    },
  );
}

export async function sendPaxMonGetTripLoadInfosRequest(
  content: PaxMonGetTripLoadInfosRequest,
): Promise<PaxMonGetTripLoadInfosResponse> {
  const msg = await sendRequest(
    "/paxmon/trip_load_info",
    "PaxMonGetTripLoadInfosRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetTripLoadInfosResponse");
  return msg.content as PaxMonGetTripLoadInfosResponse;
}

export async function sendPaxMonFindTripsRequest(
  content: PaxMonFindTripsRequest,
): Promise<PaxMonFindTripsResponse> {
  const msg = await sendRequest(
    "/paxmon/find_trips",
    "PaxMonFindTripsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonFindTripsResponse");
  return msg.content as PaxMonFindTripsResponse;
}

export function usePaxMonFindTripsQuery(
  universe: number,
  trainNr: number | undefined,
  options?: Pick<UseQueryOptions, "keepPreviousData">,
): UseQueryResult<PaxMonFindTripsResponse> {
  return useQuery(
    queryKeys.findTrips(universe, trainNr),
    () =>
      sendPaxMonFindTripsRequest({
        universe,
        train_nr: trainNr ?? 0,
        only_trips_with_paxmon_data: true,
        filter_class: false,
        max_class: 0,
      }),
    { ...options, enabled: trainNr != undefined && !isNaN(trainNr) },
  );
}

export async function sendPaxMonGroupsInTripRequest(
  content: PaxMonGetGroupsInTripRequest,
): Promise<PaxMonGetGroupsInTripResponse> {
  const msg = await sendRequest(
    "/paxmon/groups_in_trip",
    "PaxMonGetGroupsInTripRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetGroupsInTripResponse");
  return msg.content as PaxMonGetGroupsInTripResponse;
}

export function usePaxMonGroupsInTripQuery(
  content: PaxMonGetGroupsInTripRequest,
): UseQueryResult<PaxMonGetGroupsInTripResponse> {
  return useQuery(queryKeys.tripGroups(content), () =>
    sendPaxMonGroupsInTripRequest(content),
  );
}

export async function sendPaxMonAddressableGroupsRequest(
  content: PaxMonGetAddressableGroupsRequest,
): Promise<PaxMonGetAddressableGroupsResponse> {
  const msg = await sendRequest(
    "/paxmon/addressable_groups",
    "PaxMonGetAddressableGroupsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetAddressableGroupsResponse");
  return msg.content as PaxMonGetAddressableGroupsResponse;
}

export function usePaxMonAddressableGroupsQuery(
  content: PaxMonGetAddressableGroupsRequest,
): UseQueryResult<PaxMonGetAddressableGroupsResponse> {
  return useQuery(queryKeys.addressableGroups(content), () =>
    sendPaxMonAddressableGroupsRequest(content),
  );
}

export async function sendPaxMonForkUniverseRequest(
  content: PaxMonForkUniverseRequest,
): Promise<PaxMonForkUniverseResponse> {
  const msg = await sendRequest(
    "/paxmon/fork_universe",
    "PaxMonForkUniverseRequest",
    content,
  );
  verifyContentType(msg, "PaxMonForkUniverseResponse");
  return msg.content as PaxMonForkUniverseResponse;
}

export async function sendPaxMonDestroyUniverseRequest(
  content: PaxMonDestroyUniverseRequest,
): Promise<MotisSuccess> {
  const msg = await sendRequest(
    "/paxmon/destroy_universe",
    "PaxMonDestroyUniverseRequest",
    content,
  );
  verifyContentType(msg, "MotisSuccess");
  return msg.content as MotisSuccess;
}

export async function sendPaxMonGetInterchangesRequest(
  content: PaxMonGetInterchangesRequest,
): Promise<PaxMonGetInterchangesResponse> {
  const msg = await sendRequest(
    "/paxmon/get_interchanges",
    "PaxMonGetInterchangesRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetInterchangesResponse");
  return msg.content as PaxMonGetInterchangesResponse;
}

export function usePaxMonGetInterchangesQuery(
  content: PaxMonGetInterchangesRequest,
): UseQueryResult<PaxMonGetInterchangesResponse> {
  return useQuery(queryKeys.interchanges(content), () =>
    sendPaxMonGetInterchangesRequest(content),
  );
}

export async function sendPaxMonFilterTripsRequest(
  content: PaxMonFilterTripsRequest,
): Promise<PaxMonFilterTripsResponse> {
  const msg = await sendRequest(
    "/paxmon/filter_trips",
    "PaxMonFilterTripsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonFilterTripsResponse");
  return msg.content as PaxMonFilterTripsResponse;
}

export function usePaxMonFilterTripsRequest(
  content: PaxMonFilterTripsRequest,
): UseQueryResult<PaxMonFilterTripsResponse> {
  return useQuery(queryKeys.filterTrips(content), () =>
    sendPaxMonFilterTripsRequest(content),
  );
}

export async function sendPaxMonFilterGroupsRequest(
  content: PaxMonFilterGroupsRequest,
): Promise<PaxMonFilterGroupsResponse> {
  const msg = await sendRequest(
    "/paxmon/filter_groups",
    "PaxMonFilterGroupsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonFilterGroupsResponse");
  return msg.content as PaxMonFilterGroupsResponse;
}

export function usePaxMonFilterGroupsRequest(
  content: PaxMonFilterGroupsRequest,
): UseQueryResult<PaxMonFilterGroupsResponse> {
  return useQuery(queryKeys.filterGroups(content), () =>
    sendPaxMonFilterGroupsRequest(content),
  );
}

export async function sendPaxMonGetGroupsRequest(
  content: PaxMonGetGroupsRequest,
): Promise<PaxMonGetGroupsResponse> {
  const msg = await sendRequest(
    "/paxmon/get_groups",
    "PaxMonGetGroupsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetGroupsResponse");
  return msg.content as PaxMonGetGroupsResponse;
}

export function usePaxMonGetGroupsRequest(
  content: PaxMonGetGroupsRequest,
): UseQueryResult<PaxMonGetGroupsResponse> {
  return useQuery(queryKeys.getGroups(content), () =>
    sendPaxMonGetGroupsRequest(content),
  );
}

export async function sendPaxMonKeepAliveRequest(
  content: PaxMonKeepAliveRequest,
): Promise<PaxMonKeepAliveResponse> {
  const msg = await sendRequest(
    "/paxmon/keep_alive",
    "PaxMonKeepAliveRequest",
    content,
  );
  verifyContentType(msg, "PaxMonKeepAliveResponse");
  return msg.content as PaxMonKeepAliveResponse;
}

export async function sendPaxMonGroupStatisticsRequest(
  content: PaxMonGroupStatisticsRequest,
): Promise<PaxMonGroupStatisticsResponse> {
  const msg = await sendRequest(
    "/paxmon/group_statistics",
    "PaxMonGroupStatisticsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGroupStatisticsResponse");
  return msg.content as PaxMonGroupStatisticsResponse;
}

export function usePaxMonGroupStatisticsQuery(
  content: PaxMonGroupStatisticsRequest,
): UseQueryResult<PaxMonGroupStatisticsResponse> {
  return useQuery(queryKeys.groupStatistics(content), () =>
    sendPaxMonGroupStatisticsRequest(content),
  );
}

export async function sendPaxMonDebugGraphRequest(
  content: PaxMonDebugGraphRequest,
): Promise<PaxMonDebugGraphResponse> {
  const msg = await sendRequest(
    "/paxmon/debug_graph",
    "PaxMonDebugGraphRequest",
    content,
  );
  verifyContentType(msg, "PaxMonDebugGraphResponse");
  return msg.content as PaxMonDebugGraphResponse;
}

export async function sendPaxMonGetUniversesRequest(): Promise<PaxMonGetUniversesResponse> {
  const msg = await sendRequest("/paxmon/universes");
  verifyContentType(msg, "PaxMonGetUniversesResponse");
  return msg.content as PaxMonGetUniversesResponse;
}

export async function sendPaxMonGetTripCapacityRequest(
  content: PaxMonGetTripCapacityRequest,
): Promise<PaxMonGetTripCapacityResponse> {
  const msg = await sendRequest(
    "/paxmon/trip_capacity",
    "PaxMonGetTripCapacityRequest",
    content,
  );
  verifyContentType(msg, "PaxMonGetTripCapacityResponse");
  return msg.content as PaxMonGetTripCapacityResponse;
}

export function usePaxMonGetTripCapacity(
  content: PaxMonGetTripCapacityRequest,
): UseQueryResult<PaxMonGetTripCapacityResponse> {
  return useQuery(queryKeys.tripCapacity(content), () =>
    sendPaxMonGetTripCapacityRequest(content),
  );
}

export async function sendPaxMonCapacityStatusRequest(
  content: PaxMonCapacityStatusRequest,
): Promise<PaxMonCapacityStatusResponse> {
  const msg = await sendRequest(
    "/paxmon/capacity_status",
    "PaxMonCapacityStatusRequest",
    content,
  );
  verifyContentType(msg, "PaxMonCapacityStatusResponse");
  return msg.content as PaxMonCapacityStatusResponse;
}

export async function sendPaxMonDetailedCapacityStatusRequest(
  content: PaxMonDetailedCapacityStatusRequest,
): Promise<PaxMonDetailedCapacityStatusResponse> {
  const msg = await sendRequest(
    "/paxmon/detailed_capacity_status",
    "PaxMonDetailedCapacityStatusRequest",
    content,
  );
  verifyContentType(msg, "PaxMonDetailedCapacityStatusResponse");
  return msg.content as PaxMonDetailedCapacityStatusResponse;
}

export async function sendPaxMonMetricsRequest(
  content: PaxMonMetricsRequest,
): Promise<PaxMonMetricsResponse> {
  const msg = await sendRequest(
    "/paxmon/metrics",
    "PaxMonMetricsRequest",
    content,
  );
  verifyContentType(msg, "PaxMonMetricsResponse");
  return msg.content as PaxMonMetricsResponse;
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
  filterGroups: (req: PaxMonFilterGroupsRequest) =>
    [...queryKeys.all, "filter_groups", req] as const,
  getGroups: (req: PaxMonGetGroupsRequest) =>
    [...queryKeys.all, "get_groups", req] as const,
  addressableGroups: (req: PaxMonGetAddressableGroupsRequest) =>
    [...queryKeys.all, "addressable_groups", req] as const,
  keepAlive: (req: PaxMonKeepAliveRequest) =>
    [...queryKeys.all, "keep_alive", req] as const,
  groupStatistics: (req: PaxMonGroupStatisticsRequest) =>
    [...queryKeys.all, "group_statistics", req] as const,
  debugGraph: (req: PaxMonDebugGraphRequest) =>
    [...queryKeys.all, "debug_graph", req] as const,
  tripCapacity: (req: PaxMonGetTripCapacityRequest) =>
    [...queryKeys.all, "trip_capacity", req] as const,
  capacityStatus: (req: PaxMonCapacityStatusRequest) =>
    [...queryKeys.all, "capacity_status", req] as const,
  detailedCapacityStatus: (req: PaxMonDetailedCapacityStatusRequest) =>
    [...queryKeys.all, "detailed_capacity_status", req] as const,
  metrics: (req: PaxMonMetricsRequest) =>
    [...queryKeys.all, "metrics", req] as const,
};
