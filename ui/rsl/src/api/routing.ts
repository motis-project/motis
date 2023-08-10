import { verifyContentType } from "@/api/protocol/checks";
import { RoutingRequest, RoutingResponse } from "@/api/protocol/motis/routing";

import { sendRequest } from "@/api/request";

export async function sendRoutingRequest(
  content: RoutingRequest,
): Promise<RoutingResponse> {
  const msg = await sendRequest("/routing", "RoutingRequest", content);
  verifyContentType(msg, "RoutingResponse");
  return msg.content as RoutingResponse;
}
