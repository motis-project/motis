import { Message, MsgContentType } from "@/api/protocol/motis";

export function verifyContentType(
  msg: Message,
  expectedType: MsgContentType,
): Message {
  if (msg.content_type !== expectedType) {
    throw new Error(
      `Unexpected content type: ${msg.content_type}, expected ${expectedType}`,
    );
  }
  return msg;
}
