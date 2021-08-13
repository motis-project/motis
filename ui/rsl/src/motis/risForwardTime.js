import { sendRequest } from "./api";

export function sendRISForwardTimeRequest(newTime) {
  return sendRequest("/ris/forward", "RISForwardTimeRequest", {
    new_time: typeof newTime === "number" ? newTime : newTime.getTime() / 1000,
  });
}
