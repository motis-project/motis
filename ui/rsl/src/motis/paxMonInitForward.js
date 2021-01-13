import { sendRequest } from "./api";

export function sendPaxMonInitForward() {
  return sendRequest("/paxmon/init_forward", "MotisNoMessage");
}
