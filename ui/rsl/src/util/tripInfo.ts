import { TripId } from "@/api/protocol/motis";
import { PaxMonTripLoadInfo } from "@/api/protocol/motis/paxmon";

import { sendPaxMonTripLoadInfosRequest } from "@/api/paxmon";

// TODO: REMOVE

export async function loadAndProcessTripInfos(
  universe: number,
  trips: TripId[]
): Promise<PaxMonTripLoadInfo[]> {
  const res = await sendPaxMonTripLoadInfosRequest({
    universe,
    trips,
  });
  return res.load_infos;
}

export async function loadAndProcessTripInfo(
  universe: number,
  trip: TripId
): Promise<PaxMonTripLoadInfo> {
  return (await loadAndProcessTripInfos(universe, [trip]))[0];
}
