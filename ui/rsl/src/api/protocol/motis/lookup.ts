// generated file - do not modify - run update-protocol to update
import {
  Connection,
  EventType,
  Interval,
  Position,
  Station,
  TripId,
} from "@/api/protocol/motis";
import { RiBasisFahrt } from "@/api/protocol/motis/ribasis";

// lookup/LookupGeoStationIdRequest.fbs
export interface LookupGeoStationIdRequest {
  station_id: string;
  min_radius: number; // default: 0
  max_radius: number;
}

// lookup/LookupGeoStationRequest.fbs
export interface LookupGeoStationRequest {
  pos: Position;
  min_radius: number; // default: 0
  max_radius: number;
}

// lookup/LookupGeoStationRequest.fbs
export interface LookupBatchGeoStationRequest {
  requests: LookupGeoStationRequest[];
}

// lookup/LookupGeoStationResponse.fbs
export interface LookupGeoStationResponse {
  stations: Station[];
}

// lookup/LookupGeoStationResponse.fbs
export interface LookupBatchGeoStationResponse {
  responses: LookupGeoStationResponse[];
}

// lookup/LookupIdTrainRequest.fbs
export interface LookupIdTrainRequest {
  trip_id: TripId;
}

// lookup/LookupIdTrainResponse.fbs
export interface LookupIdTrainResponse {
  train: Connection;
}

// lookup/LookupMetaStationRequest.fbs
export interface LookupMetaStationRequest {
  station_id: string;
}

// lookup/LookupMetaStationRequest.fbs
export interface LookupBatchMetaStationRequest {
  requests: LookupMetaStationRequest[];
}

// lookup/LookupMetaStationResponse.fbs
export interface LookupMetaStationResponse {
  equivalent: Station[];
}

// lookup/LookupMetaStationResponse.fbs
export interface LookupBatchMetaStationResponse {
  responses: LookupMetaStationResponse[];
}

// lookup/LookupRiBasisRequest.fbs
export interface LookupRiBasisRequest {
  trip_id: TripId;
  schedule: number;
}

// lookup/LookupRiBasisResponse.fbs
export interface RiBasisTrip {
  trip_id: TripId;
  fahrt: RiBasisFahrt;
}

// lookup/LookupRiBasisResponse.fbs
export interface LookupRiBasisResponse {
  trips: RiBasisTrip[];
}

// lookup/LookupScheduleInfoResponse.fbs
export interface LookupScheduleInfoResponse {
  name: string;
  begin: number;
  end: number;
}

// lookup/LookupStationEventsRequest.fbs
export type TableType = "BOTH" | "ONLY_ARRIVALS" | "ONLY_DEPARTURES";

// lookup/LookupStationEventsRequest.fbs
export interface LookupStationEventsRequest {
  station_id: string;
  interval: Interval;
  type: TableType; // default: BOTH
}

// lookup/LookupStationEventsResponse.fbs
export interface StationEvent {
  trip_id: TripId[];
  type: EventType;
  train_nr: number;
  line_id: string;
  time: number;
  schedule_time: number;
  direction: string;
  service_name: string;
  track: string;
}

// lookup/LookupStationEventsResponse.fbs
export interface LookupStationEventsResponse {
  events: StationEvent[];
}
