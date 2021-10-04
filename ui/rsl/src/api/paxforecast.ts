import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { PaxForecastApplyMeasuresRequest } from "./protocol/motis/paxforecast";
import { MotisSuccess } from "./protocol/motis";

export async function sendPaxForecastApplyMeasuresRequest(
  content: PaxForecastApplyMeasuresRequest
): Promise<MotisSuccess> {
  const msg = await sendRequest(
    "/paxforecast/apply_measures",
    "PaxForecastApplyMeasuresRequest",
    content
  );
  verifyContentType(msg, "MotisSuccess");
  return msg.content as MotisSuccess;
}
