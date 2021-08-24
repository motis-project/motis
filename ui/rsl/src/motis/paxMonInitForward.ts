import { sendRequest } from "./api";

export function sendPaxMonInitForward(): Promise<Response> {
  return sendRequest("/paxmon/init_forward", "MotisNoMessage");
}
