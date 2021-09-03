import React from "react";
import {
  QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "react-query";

import { formatDateTime } from "./util/dateFormat";
import { sendPaxMonStatusRequest } from "./api/paxmon";
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

  const {
    data: status,
    isLoading,
    error,
  } = useQuery("status", sendPaxMonStatusRequest, {
    refetchInterval: 30 * 1000,
    refetchOnWindowFocus: true,
    staleTime: 0,
  });

  const forwardMutation = useMutation((forwardBy: number) => {
    return forwardTimeByStepped(
      queryClient,
      status?.system_time || 0,
      forwardBy
    );
  });

  const forwardInProgress = forwardMutation.isLoading;

  const buttonClass = `bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl ${
    forwardInProgress ? "text-gray-300" : ""
  }`;

  return (
    <div className="flex flex-row items-center space-x-2 m-2">
      {status ? (
        <>
          <div>System time: {formatDateTime(status.system_time)}</div>
          {[1, 5, 10, 30].map((min) => (
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
        <div>Error: {error}</div>
      )}
    </div>
  );
}

export default TimeControl;
