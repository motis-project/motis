import { UseQueryResult, useQuery } from "@tanstack/react-query";

import { verifyContentType } from "@/api/protocol/checks";
import {
  LookupRiBasisRequest,
  LookupRiBasisResponse,
  LookupScheduleInfoResponse,
} from "@/api/protocol/motis/lookup";

import { sendRequest } from "@/api/request";

export async function sendLookupScheduleInfoRequest(): Promise<LookupScheduleInfoResponse> {
  const msg = await sendRequest("/lookup/schedule_info");
  verifyContentType(msg, "LookupScheduleInfoResponse");
  return msg.content as LookupScheduleInfoResponse;
}

export function useLookupScheduleInfoQuery(): UseQueryResult<LookupScheduleInfoResponse> {
  return useQuery(queryKeys.scheduleInfo(), sendLookupScheduleInfoRequest);
}

export async function sendLookupRiBasisRequest(
  content: LookupRiBasisRequest,
): Promise<LookupRiBasisResponse> {
  const msg = await sendRequest(
    "/lookup/ribasis",
    "LookupRiBasisRequest",
    content,
  );
  verifyContentType(msg, "LookupRiBasisResponse");
  return msg.content as LookupRiBasisResponse;
}

export function useLookupRiBasisQuery(
  content: LookupRiBasisRequest,
): UseQueryResult<LookupRiBasisResponse> {
  return useQuery(queryKeys.riBasis(content), () =>
    sendLookupRiBasisRequest(content),
  );
}

export const queryKeys = {
  all: ["lookup"] as const,
  scheduleInfo: () => [...queryKeys.all, "schedule_info"] as const,
  riBasis: (req: LookupRiBasisRequest) =>
    [...queryKeys.all, "ribasis", req] as const,
};
