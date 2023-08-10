import { Message, MsgContent, MsgContentType } from "@/api/protocol/motis";

import { getApiEndpoint } from "@/api/endpoint";

export function makeMessage(
  target: string,
  contentType: MsgContentType,
  content: MsgContent,
  id = 0,
): Message {
  return {
    destination: { type: "Module", target },
    content_type: contentType,
    content,
    id,
  };
}

export function sendMessage(msg: Message): Promise<Message> {
  return fetch(`${getApiEndpoint()}?${msg.destination.target}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(msg, null, 2),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`API call failed with status ${response.status}`);
      } else {
        return response.json();
      }
    })
    .then((json) => {
      const msg = json as Message;
      if (!msg.content_type || !msg.content) {
        throw new Error(`API call returned non-message response`);
      }
      return msg;
    });
}

export function sendRequest(
  target: string,
  contentType: MsgContentType = "MotisNoMessage",
  content: MsgContent = {},
  id = 0,
): Promise<Message> {
  return sendMessage(makeMessage(target, contentType, content, id));
}
