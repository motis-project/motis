import {
  PaxForecastAlternativesRequest,
  PaxForecastAlternativesResponse,
} from "./protocol/motis/paxforecast";
import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";

export function sendPaxForecastAlternativesRequest(
  content: PaxForecastAlternativesRequest
): Promise<PaxForecastAlternativesResponse> {
  return sendRequest(
    "/paxforecast/get_alternatives",
    "PaxForecastAlternativesRequest",
    content
  ).then((msg) => {
    verifyContentType(msg, "PaxForecastAlternativesResponse");
    return msg.content as PaxForecastAlternativesResponse;
  });
}
