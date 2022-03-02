import { UseQueryResult, useQuery } from "react-query";

import { verifyContentType } from "@/api/protocol/checks";
import {
  StationGuesserRequest,
  StationGuesserResponse,
} from "@/api/protocol/motis/guesser";

import { sendRequest } from "@/api/request";

export async function sendStationGuesserRequest(
  content: StationGuesserRequest
): Promise<StationGuesserResponse> {
  const msg = await sendRequest("/guesser", "StationGuesserRequest", content);
  verifyContentType(msg, "StationGuesserResponse");
  return msg.content as StationGuesserResponse;
}

export function useStationGuesserQuery(
  content: StationGuesserRequest
): UseQueryResult<StationGuesserResponse> {
  return useQuery(
    queryKeys.stationGuess(content),
    () => sendStationGuesserRequest(content),
    { enabled: content.input.length >= 3 }
  );
}

export const queryKeys = {
  all: ["guesser"] as const,
  stationGuess: (req: StationGuesserRequest) =>
    [...queryKeys.all, req] as const,
};
