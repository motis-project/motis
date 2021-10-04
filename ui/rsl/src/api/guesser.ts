import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import {
  StationGuesserRequest,
  StationGuesserResponse,
} from "./protocol/motis/guesser";
import { useQuery, UseQueryResult } from "react-query";

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
    ["guesser", content],
    () => sendStationGuesserRequest(content),
    { enabled: content.input.length >= 3 }
  );
}
