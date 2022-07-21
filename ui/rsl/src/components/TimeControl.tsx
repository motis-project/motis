import {
  QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useAtom } from "jotai";
import { useAtomCallback } from "jotai/utils";
import { useCallback } from "react";

import { queryKeys, sendPaxMonStatusRequest } from "@/api/paxmon";
import { sendRISForwardTimeRequest } from "@/api/ris";

import {
  defaultUniverse,
  multiverseIdAtom,
  scheduleAtom,
  universeAtom,
  universesAtom,
} from "@/data/multiverse";

import { formatDate, formatTime } from "@/util/dateFormat";

async function forwardTimeByStepped(
  queryClient: QueryClient,
  schedule: number,
  currentTime: number,
  forwardBy: number,
  stepSize = 60
) {
  const endTime = currentTime + forwardBy;
  while (currentTime < endTime) {
    currentTime = Math.min(endTime, currentTime + stepSize);
    await sendRISForwardTimeRequest(currentTime, schedule);
    await queryClient.invalidateQueries();
  }
  return currentTime;
}

type TimeControlProps = {
  allowForwarding: boolean;
};

function TimeControl({ allowForwarding }: TimeControlProps): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [schedule] = useAtom(scheduleAtom);

  const updateMultiverseId = useAtomCallback(
    useCallback((get, set, arg: number) => {
      const currentMultiverseId = get(multiverseIdAtom);
      if (currentMultiverseId != arg) {
        set(multiverseIdAtom, arg);
        if (currentMultiverseId != 0) {
          // multiverse id changed = server restarted -> reset universes
          console.log(
            `multiverse id changed: ${currentMultiverseId} -> ${arg}`
          );
          set(universeAtom, 0);
          set(scheduleAtom, 0);
          set(universesAtom, [defaultUniverse]);
        }
      }
    }, [])
  );
  const {
    data: status,
    isLoading,
    error,
  } = useQuery(
    queryKeys.status(universe),
    () => sendPaxMonStatusRequest({ universe }),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      staleTime: 0,
      onSuccess: (data) => {
        updateMultiverseId(data.multiverse_id);
      },
    }
  );

  const forwardMutation = useMutation((forwardBy: number) => {
    return forwardTimeByStepped(
      queryClient,
      schedule,
      status?.system_time || 0,
      forwardBy
    );
  });

  const forwardDisabled = forwardMutation.isLoading;

  const buttonClass = `px-3 py-1 rounded text-sm ${
    !forwardDisabled
      ? "bg-db-red-500 hover:bg-db-red-600 text-white"
      : "bg-db-red-300 text-db-red-100 cursor-wait"
  }`;

  const buttons = allowForwarding ? (
    <>
      {[1, 10, 30].map((min) => (
        <button
          key={`${min}m`}
          type="button"
          className={buttonClass}
          disabled={forwardDisabled}
          onClick={() => {
            forwardMutation.mutate(60 * min);
          }}
        >
          +{min}m
        </button>
      ))}
      {[1, 5, 10, 12, 24].map((hrs) => (
        <button
          key={`${hrs}h`}
          type="button"
          className={buttonClass}
          disabled={forwardDisabled}
          onClick={() => {
            forwardMutation.mutate(60 * 60 * hrs);
          }}
        >
          +{hrs}h
        </button>
      ))}
    </>
  ) : null;

  return (
    <div className="flex justify-center items-baseline space-x-2">
      {status ? (
        <>
          <div>{formatDate(status.system_time)}</div>
          <div className="font-bold">{formatTime(status.system_time)}</div>
          {buttons}
        </>
      ) : isLoading ? (
        <div>Verbindung zu MOTIS wird aufgebaut...</div>
      ) : (
        <div>Fehler: {error instanceof Error ? error.message : `${error}`}</div>
      )}
    </div>
  );
}

export default TimeControl;
