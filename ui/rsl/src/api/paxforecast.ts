import { verifyContentType } from "@/api/protocol/checks";
import {
  PaxForecastApplyMeasuresRequest,
  PaxForecastApplyMeasuresResponse,
  PaxForecastMetricsRequest,
  PaxForecastMetricsResponse,
} from "@/api/protocol/motis/paxforecast";

import { sendRequest } from "@/api/request";

export async function sendPaxForecastApplyMeasuresRequest(
  content: PaxForecastApplyMeasuresRequest,
): Promise<PaxForecastApplyMeasuresResponse> {
  const msg = await sendRequest(
    "/paxforecast/apply_measures",
    "PaxForecastApplyMeasuresRequest",
    content,
  );
  verifyContentType(msg, "PaxForecastApplyMeasuresResponse");
  return msg.content as PaxForecastApplyMeasuresResponse;
}

export async function sendPaxForecastMetricsRequest(
  content: PaxForecastMetricsRequest,
): Promise<PaxForecastMetricsResponse> {
  const msg = await sendRequest(
    "/paxforecast/metrics",
    "PaxForecastMetricsRequest",
    content,
  );
  verifyContentType(msg, "PaxForecastMetricsResponse");
  return msg.content as PaxForecastMetricsResponse;
}

export const queryKeys = {
  all: ["paxforecast"] as const,
  metrics: (req: PaxForecastMetricsRequest) =>
    [...queryKeys.all, "metrics", req] as const,
};
