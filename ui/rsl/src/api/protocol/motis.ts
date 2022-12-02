// generated file - do not modify - run update-protocol to update
import {
  StationGuesserRequest,
  StationGuesserResponse,
} from "@/api/protocol/motis/guesser";
import {
  LookupBatchGeoStationRequest,
  LookupBatchGeoStationResponse,
  LookupBatchMetaStationRequest,
  LookupBatchMetaStationResponse,
  LookupGeoStationIdRequest,
  LookupGeoStationRequest,
  LookupGeoStationResponse,
  LookupIdTrainRequest,
  LookupIdTrainResponse,
  LookupMetaStationRequest,
  LookupMetaStationResponse,
  LookupRiBasisRequest,
  LookupRiBasisResponse,
  LookupScheduleInfoResponse,
  LookupStationEventsRequest,
  LookupStationEventsResponse,
} from "@/api/protocol/motis/lookup";
import {
  PaxForecastApplyMeasuresRequest,
  PaxForecastApplyMeasuresResponse,
  PaxForecastUpdate,
} from "@/api/protocol/motis/paxforecast";
import {
  PaxMonAddGroupsRequest,
  PaxMonAddGroupsResponse,
  PaxMonDebugGraphRequest,
  PaxMonDebugGraphResponse,
  PaxMonDestroyUniverseRequest,
  PaxMonFilterGroupsRequest,
  PaxMonFilterGroupsResponse,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsResponse,
  PaxMonFindTripsRequest,
  PaxMonFindTripsResponse,
  PaxMonForkUniverseRequest,
  PaxMonForkUniverseResponse,
  PaxMonGetAddressableGroupsRequest,
  PaxMonGetAddressableGroupsResponse,
  PaxMonGetGroupsInTripRequest,
  PaxMonGetGroupsInTripResponse,
  PaxMonGetGroupsRequest,
  PaxMonGetGroupsResponse,
  PaxMonGetInterchangesRequest,
  PaxMonGetInterchangesResponse,
  PaxMonGetTripLoadInfosRequest,
  PaxMonGetTripLoadInfosResponse,
  PaxMonGetUniversesResponse,
  PaxMonGroupStatisticsRequest,
  PaxMonGroupStatisticsResponse,
  PaxMonKeepAliveRequest,
  PaxMonKeepAliveResponse,
  PaxMonRemoveGroupsRequest,
  PaxMonRerouteGroupsRequest,
  PaxMonRerouteGroupsResponse,
  PaxMonStatusRequest,
  PaxMonStatusResponse,
  PaxMonTripLoadInfo,
  PaxMonUniverseDestroyed,
  PaxMonUniverseForked,
  PaxMonUpdate,
} from "@/api/protocol/motis/paxmon";
import { RISForwardTimeRequest } from "@/api/protocol/motis/ris";
import { RoutingRequest, RoutingResponse } from "@/api/protocol/motis/routing";

// base/Connection.fbs
export interface EventInfo {
  time: number;
  schedule_time: number;
  track: string;
  schedule_track: string;
  valid: boolean;
  reason: TimestampReason;
}

// base/Connection.fbs
export interface Stop {
  station: Station;
  arrival: EventInfo;
  departure: EventInfo;
  exit: boolean;
  enter: boolean;
}

// base/Connection.fbs
export interface Range {
  from: number;
  to: number;
}

// base/Connection.fbs
export interface Transport {
  range: Range;
  category_name: string;
  category_id: number;
  clasz: number;
  train_nr: number;
  line_id: string;
  name: string;
  provider: string;
  direction: string;
}

// base/Connection.fbs
export interface Walk {
  range: Range;
  mumo_id: number;
  price: number;
  accessibility: number;
  mumo_type: string;
}

// base/Connection.fbs
export type Move = Transport | Walk;

export type MoveType = "Transport" | "Walk";

// base/Connection.fbs
export interface MoveWrapper {
  move_type: MoveType;
  move: Move;
}

// base/Connection.fbs
export interface Attribute {
  range: Range;
  code: string;
  text: string;
}

// base/Connection.fbs
export interface Trip {
  range: Range;
  id: TripId;
  debug: string;
}

// base/Connection.fbs
export interface FreeText {
  range: Range;
  code: number;
  text: string;
  type: string;
}

// base/Connection.fbs
export interface Problem {
  range: Range;
  type: ProblemType;
}

// base/Connection.fbs
export interface Connection {
  stops: Stop[];
  transports: MoveWrapper[];
  trips: Trip[];
  attributes: Attribute[];
  free_texts: FreeText[];
  problems: Problem[];
  night_penalty: number;
  db_costs: number;
  status: ConnectionStatus;
}

// base/ConnectionStatus.fbs
export type ConnectionStatus =
  | "OK"
  | "INTERCHANGE_INVALID"
  | "TRAIN_HAS_BEEN_CANCELED"
  | "INVALID";

// base/DirectConnection.fbs
export interface DirectConnection {
  duration: number;
  accessibility: number;
  mumo_type: string;
}

// base/EventType.fbs
export type EventType = "DEP" | "ARR";

// base/Interval.fbs
export interface Interval {
  begin: number;
  end: number;
}

// base/Polyline.fbs
export interface Polyline {
  coordinates: number[];
}

// base/Position.fbs
export interface Position {
  lat: number;
  lng: number;
}

// base/ProblemType.fbs
export type ProblemType =
  | "NO_PROBLEM"
  | "INTERCHANGE_TIME_VIOLATED"
  | "CANCELED_TRAIN";

// base/SearchDir.fbs
export type SearchDir = "Forward" | "Backward";

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

// base/Statistics.fbs
export interface Statistics {
  category: string; // key
  entries: StatisticsEntry[];
}

// base/Statistics.fbs
export interface StatisticsEntry {
  name: string; // key
  value: number;
}

// base/TimestampReason.fbs
export type TimestampReason =
  | "SCHEDULE"
  | "REPAIR"
  | "IS"
  | "PROPAGATION"
  | "FORECAST";

// base/TripId.fbs
export interface TripId {
  station_id: string;
  train_nr: number;
  time: number;
  target_station_id: string;
  target_time: number;
  line_id: string;
}

// base/TripInfo.fbs
export interface TripInfo {
  id: TripId;
  transport: Transport;
}

// base/TripServiceInfo.fbs
export interface TripServiceInfo {
  trip: TripId;
  primary_station: Station;
  secondary_station: Station;
  service_infos: ServiceInfo[];
}

// HTTPMessage.fbs
export interface HTTPHeader {
  name: string;
  value: string;
}

// HTTPMessage.fbs
export type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE" | "OPTIONS";

// HTTPMessage.fbs
export type HTTPStatus = "OK" | "INTERNAL_SERVER_ERROR";

// HTTPMessage.fbs
export interface HTTPRequest {
  method: HTTPMethod;
  path: string;
  headers: HTTPHeader[];
  content: string;
}

// HTTPMessage.fbs
export interface HTTPResponse {
  status: HTTPStatus;
  headers: HTTPHeader[];
  content: string;
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
export interface ApiDescription {
  methods: string[];
}

// Message.fbs
export type MsgContent =
  | MotisNoMessage
  | MotisError
  | MotisSuccess
  | HTTPRequest
  | HTTPResponse
  | ApiDescription
  | Connection
  | TripId
  | StationGuesserRequest
  | StationGuesserResponse
  | LookupGeoStationIdRequest
  | LookupGeoStationRequest
  | LookupGeoStationResponse
  | LookupBatchGeoStationRequest
  | LookupBatchGeoStationResponse
  | LookupStationEventsRequest
  | LookupStationEventsResponse
  | LookupScheduleInfoResponse
  | LookupMetaStationRequest
  | LookupMetaStationResponse
  | LookupBatchMetaStationRequest
  | LookupBatchMetaStationResponse
  | LookupIdTrainRequest
  | LookupIdTrainResponse
  | RISForwardTimeRequest
  | RoutingRequest
  | RoutingResponse
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
  | PaxMonUniverseForked
  | PaxMonUniverseDestroyed
  | PaxMonGetInterchangesRequest
  | PaxMonGetInterchangesResponse
  | PaxMonStatusRequest
  | LookupRiBasisRequest
  | LookupRiBasisResponse
  | PaxForecastApplyMeasuresResponse
  | PaxMonGetAddressableGroupsRequest
  | PaxMonGetAddressableGroupsResponse
  | PaxMonKeepAliveRequest
  | PaxMonKeepAliveResponse
  | PaxMonRerouteGroupsRequest
  | PaxMonRerouteGroupsResponse
  | PaxMonGroupStatisticsRequest
  | PaxMonGroupStatisticsResponse
  | PaxMonDebugGraphRequest
  | PaxMonDebugGraphResponse
  | PaxMonGetUniversesResponse;

export type MsgContentType =
  | "MotisNoMessage"
  | "MotisError"
  | "MotisSuccess"
  | "HTTPRequest"
  | "HTTPResponse"
  | "ApiDescription"
  | "Connection"
  | "TripId"
  | "StationGuesserRequest"
  | "StationGuesserResponse"
  | "LookupGeoStationIdRequest"
  | "LookupGeoStationRequest"
  | "LookupGeoStationResponse"
  | "LookupBatchGeoStationRequest"
  | "LookupBatchGeoStationResponse"
  | "LookupStationEventsRequest"
  | "LookupStationEventsResponse"
  | "LookupScheduleInfoResponse"
  | "LookupMetaStationRequest"
  | "LookupMetaStationResponse"
  | "LookupBatchMetaStationRequest"
  | "LookupBatchMetaStationResponse"
  | "LookupIdTrainRequest"
  | "LookupIdTrainResponse"
  | "RISForwardTimeRequest"
  | "RoutingRequest"
  | "RoutingResponse"
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
  | "PaxMonUniverseForked"
  | "PaxMonUniverseDestroyed"
  | "PaxMonGetInterchangesRequest"
  | "PaxMonGetInterchangesResponse"
  | "PaxMonStatusRequest"
  | "LookupRiBasisRequest"
  | "LookupRiBasisResponse"
  | "PaxForecastApplyMeasuresResponse"
  | "PaxMonGetAddressableGroupsRequest"
  | "PaxMonGetAddressableGroupsResponse"
  | "PaxMonKeepAliveRequest"
  | "PaxMonKeepAliveResponse"
  | "PaxMonRerouteGroupsRequest"
  | "PaxMonRerouteGroupsResponse"
  | "PaxMonGroupStatisticsRequest"
  | "PaxMonGroupStatisticsResponse"
  | "PaxMonDebugGraphRequest"
  | "PaxMonDebugGraphResponse"
  | "PaxMonGetUniversesResponse";

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
