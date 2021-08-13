import { sendRequest } from "./api";

export function sendPaxMonStatusRequest() {
  return sendRequest("/paxmon/status", "MotisNoMessage");
}
