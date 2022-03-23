import { TripId } from "@/api/protocol/motis";

import { sendPaxMonTripLoadInfosRequest } from "@/api/paxmon";

import { PaxMonTripLoadInfoWithStats } from "@/data/loadInfo";

import { addEdgeStatistics } from "@/util/statistics";

export async function loadAndProcessTripInfos(
  universe: number,
  trips: TripId[]
): Promise<PaxMonTripLoadInfoWithStats[]> {
  const res = await sendPaxMonTripLoadInfosRequest({
    universe,
    trips,
  });
  return res.load_infos.map(addEdgeStatistics);
}

export async function loadAndProcessTripInfo(
  universe: number,
  trip: TripId
): Promise<PaxMonTripLoadInfoWithStats> {
  return (await loadAndProcessTripInfos(universe, [trip]))[0];
}
