import { sendRequest } from "./request";

export function sendRISForwardTimeRequest(
  newTime: number | Date
): Promise<Response> {
  return sendRequest("/ris/forward", "RISForwardTimeRequest", {
    new_time: typeof newTime === "number" ? newTime : newTime.getTime() / 1000,
  });
}
