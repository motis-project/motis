import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAtom } from "jotai";
import React from "react";

import { TripId } from "@/api/protocol/motis";

import { queryKeys, sendPaxMonGetTripLoadInfosRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import { formatDateTime } from "@/util/dateFormat";

import MiniTripLoadGraph from "@/components/trips/MiniTripLoadGraph";

export interface TripTooltipProps {
  tripId: TripId;
}

export function TripTooltip({ tripId }: TripTooltipProps): JSX.Element | null {
  const [universe] = useAtom(universeAtom);

  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery(
    queryKeys.tripLoad(universe, tripId),
    () => sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    {
      placeholderData: () => {
        return universe != 0
          ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
          : undefined;
      },
    },
  );

  if (!data || data.load_infos.length === 0) {
    return null;
  }

  const li = data.load_infos[0];

  const category = li.tsi.service_infos[0]?.category ?? "";
  const trainNr = li.tsi.service_infos[0]?.train_nr ?? li.tsi.trip.train_nr;

  return (
    <div className="w-[30rem] bg-white p-2 rounded-md shadow-lg flex flex-col gap-2">
      <div className="flex gap-4 pb-1">
        <div className="flex flex-col">
          <div className="text-sm text-center">{category}</div>
          <div className="text-xl font-semibold">{trainNr}</div>
        </div>
        <div className="grow flex flex-col truncate">
          <div className="flex justify-between">
            <div className="truncate">{li.tsi.primary_station.name}</div>
            <div>{formatDateTime(li.tsi.trip.time)}</div>
          </div>
          <div className="flex justify-between">
            <div className="truncate">{li.tsi.secondary_station.name}</div>
            <div>{formatDateTime(li.tsi.trip.target_time)}</div>
          </div>
        </div>
      </div>
      <MiniTripLoadGraph edges={li.edges} />
    </div>
  );
}
