import { useAtom } from "jotai";
import { QueryClient, useMutation, useQueryClient } from "react-query";

import { usePaxMonStatusQuery } from "@/api/paxmon";
import { sendRISForwardTimeRequest } from "@/api/ris";

import { scheduleAtom, universeAtom } from "@/data/simulation";

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
  const { data: status, isLoading, error } = usePaxMonStatusQuery(universe);

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
        <div>Fehler: {error instanceof Error ? error.message : error}</div>
      )}
    </div>
  );
}

export default TimeControl;
