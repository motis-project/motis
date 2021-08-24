import { sendRequest } from "./api";

export function sendPaxMonStatusRequest(): Promise<Response> {
  return sendRequest({
    destination: { target: "/paxmon/status" },
    content_type: "MotisNoMessage",
    content: {},
  });
}
