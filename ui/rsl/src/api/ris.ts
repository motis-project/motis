import { UseQueryResult, useQuery } from "@tanstack/react-query";

import { verifyContentType } from "@/api/protocol/checks";
import { Message } from "@/api/protocol/motis";
import { RISStatusResponse } from "@/api/protocol/motis/ris";

import { sendRequest } from "@/api/request";

export function sendRISForwardTimeRequest(
  newTime: number | Date,
  schedule: number,
): Promise<Message> {
  return sendRequest("/ris/forward", "RISForwardTimeRequest", {
    new_time: typeof newTime === "number" ? newTime : newTime.getTime() / 1000,
    schedule,
  }).then((msg) => {
    return verifyContentType(msg, "MotisSuccess");
  });
}

export async function sendRISStatusRequest(): Promise<RISStatusResponse> {
  const msg = await sendRequest("/ris/status");
  verifyContentType(msg, "RISStatusResponse");
  return msg.content as RISStatusResponse;
}

export function useRISStatusRequest(): UseQueryResult<RISStatusResponse> {
  return useQuery(queryKeys.status(), () => sendRISStatusRequest(), {
    refetchInterval: 30 * 1000,
    refetchOnWindowFocus: true,
    staleTime: 0,
  });
}

export const queryKeys = {
  all: ["ris"] as const,
  status: () => [...queryKeys.all, "status"] as const,
};
