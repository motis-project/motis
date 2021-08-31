// generated file - do not modify - run update-protocol to update

import { RISForwardTimeRequest } from "./motis/ris";
import {
  PaxMonUpdate,
  PaxMonAddGroupsRequest,
  PaxMonAddGroupsResponse,
  PaxMonRemoveGroupsRequest,
  PaxMonTripLoadInfo,
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonStatusResponse,
  PaxMonGetGroupsRequest,
  PaxMonGetGroupsResponse,
  PaxMonFilterGroupsRequest,
  PaxMonFilterGroupsResponse,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonForkUniverseRequest,
  PaxMonForkUniverseResponse,
  PaxMonDestroyUniverseRequest,
  PaxMonGetGroupsInTripRequest,
  PaxMonGetGroupsInTripResponse,
} from "./motis/paxmon";
import {
  PaxForecastUpdate,
  PaxForecastApplyMeasuresRequest,
  PaxForecastAlternativesRequest,
  PaxForecastAlternativesResponse,
} from "./motis/paxforecast";

// base/Interval.fbs
export interface Interval {
  begin: number;
  end: number;
}

// base/Position.fbs
export interface Position {
  lat: number;
  lng: number;
}

// base/ServiceInfo.fbs
export interface ServiceInfo {
  name: string;
  category: string;
  train_nr: number;
  line: string;
  provider: string;
  clasz: number;
}

// base/Station.fbs
export interface Station {
  id: string;
  name: string;
  pos: Position;
}

// base/TripId.fbs
export interface TripId {
  station_id: string;
  train_nr: number;
  time: number;
  target_station_id: string;
  target_time: number;
  line_id: string;
}

// base/TripServiceInfo.fbs
export interface TripServiceInfo {
  trip: TripId;
  primary_station: Station;
  secondary_station: Station;
  service_infos: ServiceInfo[];
}

// Message.fbs
export interface MotisError {
  error_code: number;
  category: string;
  reason: string;
}

// Message.fbs
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface MotisSuccess {}

// Message.fbs
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface MotisNoMessage {}

// Message.fbs
export type MsgContent =
  | MotisNoMessage
  | MotisError
  | MotisSuccess
  | TripId
  | RISForwardTimeRequest
  | PaxMonUpdate
  | PaxForecastUpdate
  | PaxMonAddGroupsRequest
  | PaxMonAddGroupsResponse
  | PaxMonRemoveGroupsRequest
  | PaxMonTripLoadInfo
  | PaxMonFindTripsRequest
  | PaxMonFindTripsResponse
  | PaxMonStatusResponse
  | PaxMonGetGroupsRequest
  | PaxMonGetGroupsResponse
  | PaxMonFilterGroupsRequest
  | PaxMonFilterGroupsResponse
  | PaxMonFilterTripsRequest
  | PaxMonFilterTripsResponse
  | PaxMonGetTripLoadInfosRequest
  | PaxMonGetTripLoadInfosResponse
  | PaxMonForkUniverseRequest
  | PaxMonForkUniverseResponse
  | PaxMonDestroyUniverseRequest
  | PaxMonGetGroupsInTripRequest
  | PaxMonGetGroupsInTripResponse
  | PaxForecastApplyMeasuresRequest
  | PaxForecastAlternativesRequest
  | PaxForecastAlternativesResponse;

export type MsgContentType =
  | "MotisNoMessage"
  | "MotisError"
  | "MotisSuccess"
  | "TripId"
  | "RISForwardTimeRequest"
  | "PaxMonUpdate"
  | "PaxForecastUpdate"
  | "PaxMonAddGroupsRequest"
  | "PaxMonAddGroupsResponse"
  | "PaxMonRemoveGroupsRequest"
  | "PaxMonTripLoadInfo"
  | "PaxMonFindTripsRequest"
  | "PaxMonFindTripsResponse"
  | "PaxMonStatusResponse"
  | "PaxMonGetGroupsRequest"
  | "PaxMonGetGroupsResponse"
  | "PaxMonFilterGroupsRequest"
  | "PaxMonFilterGroupsResponse"
  | "PaxMonFilterTripsRequest"
  | "PaxMonFilterTripsResponse"
  | "PaxMonGetTripLoadInfosRequest"
  | "PaxMonGetTripLoadInfosResponse"
  | "PaxMonForkUniverseRequest"
  | "PaxMonForkUniverseResponse"
  | "PaxMonDestroyUniverseRequest"
  | "PaxMonGetGroupsInTripRequest"
  | "PaxMonGetGroupsInTripResponse"
  | "PaxForecastApplyMeasuresRequest"
  | "PaxForecastAlternativesRequest"
  | "PaxForecastAlternativesResponse";

// Message.fbs
export type DestinationType = "Module" | "Topic";

// Message.fbs
export interface Destination {
  type: DestinationType;
  target: string;
}

// Message.fbs
export interface Message {
  destination: Destination;
  content_type: MsgContentType;
  content: MsgContent;
  id: number; // default: 0
}
