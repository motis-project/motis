import React from "react";
import { QueryClient, useMutation, useQueryClient } from "react-query";

import { formatDate, formatTime } from "./util/dateFormat";
import { usePaxMonStatusQuery } from "./api/paxmon";
import { sendRISForwardTimeRequest } from "./api/ris";

async function forwardTimeByStepped(
  queryClient: QueryClient,
  currentTime: number,
  forwardBy: number,
  stepSize = 60
) {
  const endTime = currentTime + forwardBy;
  while (currentTime < endTime) {
    currentTime = Math.min(endTime, currentTime + stepSize);
    await sendRISForwardTimeRequest(currentTime);
    await queryClient.invalidateQueries();
  }
  return currentTime;
}

function TimeControl(): JSX.Element {
  const queryClient = useQueryClient();

  const { data: status, isLoading, error } = usePaxMonStatusQuery();

  const forwardMutation = useMutation((forwardBy: number) => {
    return forwardTimeByStepped(
      queryClient,
      status?.system_time || 0,
      forwardBy
    );
  });

  const forwardInProgress = forwardMutation.isLoading;

  const buttonClass = `bg-blue-900 px-2 py-1 rounded-xl text-white text-sm ${
    forwardInProgress ? "text-blue-700" : "hover:bg-blue-800"
  }`;

  return (
    <div className="flex justify-center items-baseline space-x-2 p-2 mb-2 bg-blue-700 text-white">
      {status ? (
        <>
          <div>{formatDate(status.system_time)}</div>
          <div className="font-bold">{formatTime(status.system_time)}</div>
          {[1, 10, 30].map((min) => (
            <button
              key={`${min}m`}
              className={buttonClass}
              disabled={forwardInProgress}
              onClick={() => {
                forwardMutation.mutate(60 * min);
              }}
            >
              +{min}m
            </button>
          ))}
          {[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 24,
          ].map((hrs) => (
            <button
              key={`${hrs}h`}
              className={buttonClass}
              disabled={forwardInProgress}
              onClick={() => {
                forwardMutation.mutate(60 * 60 * hrs);
              }}
            >
              +{hrs}h
            </button>
          ))}
        </>
      ) : isLoading ? (
        <div>System time: loading...</div>
      ) : (
        <div>Error: {error instanceof Error ? error.message : error}</div>
      )}
    </div>
  );
}

export default TimeControl;
