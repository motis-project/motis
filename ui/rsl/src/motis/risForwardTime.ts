import { sendRequest } from "./api";

export function sendRISForwardTimeRequest(
  newTime: number | Date
): Promise<Response> {
  return sendRequest({
    destination: { target: "/ris/forward" },
    content_type: "RISForwardTimeRequest",
    content: {
      new_time:
        typeof newTime === "number" ? newTime : newTime.getTime() / 1000,
    },
  });
}
