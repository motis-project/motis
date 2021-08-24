import { sendRequest } from "./api";

export function sendPaxMonStatusRequest(): Promise<Response> {
  return sendRequest("/paxmon/status", "MotisNoMessage");
}
