// generated file - do not modify - run update-protocol to update
import {
  Connection,
  DirectConnection,
  Interval,
  Statistics,
  TripId,
} from "@/api/protocol/motis";

// routing/RoutingRequest.fbs
export interface InputStation {
  id: string;
  name: string;
}

// routing/RoutingRequest.fbs
export interface MumoEdge {
  from_station_id: string;
  to_station_id: string;
  duration: number;
  price: number;
  accessibility: number;
  mumo_id: number;
}

// routing/RoutingRequest.fbs
export interface PeriodicMumoEdge {
  edge: MumoEdge;
  interval: Interval;
}

// routing/RoutingRequest.fbs
export interface TimeDependentMumoEdge {
  edge: MumoEdge;
  interval: Interval;
}

// routing/RoutingRequest.fbs
export interface HotelEdge {
  edge: MumoEdge;
  earliest_checkout_time: number;
  min_stay_duration: number;
}

// routing/RoutingRequest.fbs
export type AdditionalEdge =
  | MumoEdge
  | PeriodicMumoEdge
  | TimeDependentMumoEdge
  | HotelEdge;

export type AdditionalEdgeType =
  | "MumoEdge"
  | "PeriodicMumoEdge"
  | "TimeDependentMumoEdge"
  | "HotelEdge";

// routing/RoutingRequest.fbs
export interface AdditionalEdgeWrapper {
  additional_edge_type: AdditionalEdgeType;
  additional_edge: AdditionalEdge;
}

// routing/RoutingRequest.fbs
export interface OntripTrainStart {
  trip: TripId;
  station: InputStation;
  arrival_time: number;
}

// routing/RoutingRequest.fbs
export interface OntripStationStart {
  station: InputStation;
  departure_time: number;
}

// routing/RoutingRequest.fbs
export interface PretripStart {
  station: InputStation;
  interval: Interval;
  min_connection_count: number;
  extend_interval_earlier: boolean;
  extend_interval_later: boolean;
}

// routing/RoutingRequest.fbs
export type Start = OntripTrainStart | OntripStationStart | PretripStart;

export type StartType =
  | "OntripTrainStart"
  | "OntripStationStart"
  | "PretripStart";

// routing/RoutingRequest.fbs
export interface Via {
  station: InputStation;
  stay_duration: number;
}

// routing/RoutingRequest.fbs
export type SearchType =
  | "Default"
  | "SingleCriterion"
  | "SingleCriterionNoIntercity"
  | "LateConnections"
  | "LateConnectionsTest"
  | "Accessibility"
  | "DefaultPrice"
  | "DefaultPriceRegional";

// routing/RoutingRequest.fbs
export type SearchDir = "Forward" | "Backward";

// routing/RoutingRequest.fbs
export interface RoutingRequest {
  start_type: StartType;
  start: Start;
  destination: InputStation;
  search_type: SearchType;
  search_dir: SearchDir;
  via: Via[];
  additional_edges: AdditionalEdgeWrapper[];
  use_start_metas: boolean; // default: true
  use_dest_metas: boolean; // default: true
  use_start_footpaths: boolean; // default: true
  schedule: number;
}

// routing/RoutingResponse.fbs
export interface RoutingResponse {
  statistics: Statistics[];
  connections: Connection[];
  interval_begin: number;
  interval_end: number;
  direct_connections: DirectConnection[];
}
