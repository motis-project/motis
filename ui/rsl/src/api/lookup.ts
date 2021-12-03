import { LookupScheduleInfoResponse } from "./protocol/motis/lookup";
import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { useQuery, UseQueryResult } from "react-query";

export async function sendLookupScheduleInfoRequest(): Promise<LookupScheduleInfoResponse> {
  const msg = await sendRequest("/lookup/schedule_info");
  verifyContentType(msg, "LookupScheduleInfoResponse");
  return msg.content as LookupScheduleInfoResponse;
}

export function useLookupScheduleInfoQuery(): UseQueryResult<LookupScheduleInfoResponse> {
  return useQuery(queryKeys.scheduleInfo(), sendLookupScheduleInfoRequest);
}

export const queryKeys = {
  all: ["lookup"] as const,
  scheduleInfo: () => [...queryKeys.all, "schedule_info"] as const,
};
