import { verifyContentType } from "@/api/protocol/checks";
import { Message } from "@/api/protocol/motis";

import { sendRequest } from "@/api/request";

export function sendRISForwardTimeRequest(
  newTime: number | Date,
  schedule: number
): Promise<Message> {
  return sendRequest("/ris/forward", "RISForwardTimeRequest", {
    new_time: typeof newTime === "number" ? newTime : newTime.getTime() / 1000,
    schedule,
  }).then((msg) => {
    return verifyContentType(msg, "MotisSuccess");
  });
}
