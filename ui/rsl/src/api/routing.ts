import { sendRequest } from "./request";
import { verifyContentType } from "./protocol/checks";
import { RoutingRequest, RoutingResponse } from "./protocol/motis/routing";

export async function sendRoutingRequest(
  content: RoutingRequest
): Promise<RoutingResponse> {
  const msg = await sendRequest("/routing", "RoutingRequest", content);
  verifyContentType(msg, "RoutingResponse");
  return msg.content as RoutingResponse;
}
