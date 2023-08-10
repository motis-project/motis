import { UseQueryResult, useQuery } from "@tanstack/react-query";

import { verifyContentType } from "@/api/protocol/checks";
import { RtMetricsResponse } from "@/api/protocol/motis/rt";

import { sendRequest } from "@/api/request";

export async function sendRtMetricsRequest(): Promise<RtMetricsResponse> {
  const msg = await sendRequest("/rt/metrics");
  verifyContentType(msg, "RtMetricsResponse");
  return msg.content as RtMetricsResponse;
}

export function useRtMetricsRequest(): UseQueryResult<RtMetricsResponse> {
  return useQuery(queryKeys.metrics(), () => sendRtMetricsRequest(), {
    refetchInterval: 30 * 1000,
    refetchOnWindowFocus: true,
    staleTime: 0,
  });
}

export const queryKeys = {
  all: ["rt"] as const,
  metrics: () => [...queryKeys.all, "metrics"] as const,
};
