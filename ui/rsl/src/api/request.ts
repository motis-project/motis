import { Message, MsgContent, MsgContentType } from "./protocol/motis";
import apiEndpoint from "./endpoint";

export function makeMessage(
  target: string,
  contentType: MsgContentType,
  content: MsgContent,
  id = 0
): Message {
  return {
    destination: { type: "Module", target },
    content_type: contentType,
    content,
    id,
  };
}

export function sendMessage(msg: Message): Promise<Response> {
  return fetch(`${apiEndpoint}?${msg.destination.target}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(msg, null, 2),
  });
}

export function sendRequest(
  target: string,
  contentType: MsgContentType = "MotisNoMessage",
  content: MsgContent = {},
  id = 0
): Promise<Response> {
  return sendMessage(makeMessage(target, contentType, content, id));
}
