import {
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonStatusResponse,
  PaxMonTripLoadInfo,
} from "./paxmon";
import { RISForwardTimeRequest } from "./ris";
import { TripId } from "./base";

export const enum DestinationType {
  Module,
  Topic,
}

export interface Destination {
  target: string;
  type?: DestinationType;
}

export interface MessageBase {
  destination: Destination;
  id?: number;
}

////////////////////////////////////////////////////////////////////////////////

export interface MotisError {
  error_code: number;
  category: string;
  reason: string;
}

export interface MotisErrorMessage extends MessageBase {
  content_type: "MotisError";
  content: MotisError;
}

export interface MotisSuccessMessage extends MessageBase {
  content_type: "MotisSuccess";
  content: Record<string, never>;
}

export interface MotisNoMessage extends MessageBase {
  content_type: "MotisNoMessage";
  content: Record<string, never>;
}

////////////////////////////////////////////////////////////////////////////////

export interface TripIdMessage extends MessageBase {
  content_type: "TripId";
  content: TripId;
}

////////////////////////////////////////////////////////////////////////////////

export interface PaxMonFindTripsRequestMessage extends MessageBase {
  content_type: "PaxMonFindTripsRequest";
  content: PaxMonFindTripsRequest;
}

export interface PaxMonTripLoadInfoMessage extends MessageBase {
  content_type: "PaxMonTripLoadInfo";
  content: PaxMonTripLoadInfo;
}

export interface PaxMonStatusResponseMessage extends MessageBase {
  content_type: "PaxMonStatusResponse";
  content: PaxMonStatusResponse;
}

export interface PaxMonFindTripsResponseMessage extends MessageBase {
  content_type: "PaxMonFindTripsResponse";
  content: PaxMonFindTripsResponse;
}

export interface PaxMonGetTripLoadInfosRequestMessage extends MessageBase {
  content_type: "PaxMonGetTripLoadInfosRequest";
  content: PaxMonGetTripLoadInfosRequest;
}

export interface PaxMonGetTripLoadInfosResponseMessage extends MessageBase {
  content_type: "PaxMonGetTripLoadInfosResponse";
  content: PaxMonGetTripLoadInfosResponse;
}

////////////////////////////////////////////////////////////////////////////////

export interface RISForwardTimeRequestMessage extends MessageBase {
  content_type: "RISForwardTimeRequest";
  content: RISForwardTimeRequest;
}

////////////////////////////////////////////////////////////////////////////////

export type MotisMessage =
  | MotisErrorMessage
  | MotisSuccessMessage
  | MotisNoMessage
  | TripIdMessage
  | PaxMonFindTripsRequestMessage
  | PaxMonTripLoadInfoMessage
  | PaxMonStatusResponseMessage
  | PaxMonFindTripsResponseMessage
  | PaxMonGetTripLoadInfosRequestMessage
  | PaxMonGetTripLoadInfosResponseMessage
  | RISForwardTimeRequestMessage;
