import { sendRequest } from "./api";

export function sendPaxMonInitForward(): Promise<Response> {
  return sendRequest({
    destination: { target: "/paxmon/init_forward" },
    content_type: "MotisNoMessage",
    content: {},
  });
}
