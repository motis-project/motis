import { Station, TripServiceInfo } from "./base";

export interface PaxMonFindTripsRequest {
  universe?: number;
  train_nr: number;
  only_trips_with_paxmon_data?: boolean;
  filter_class?: boolean;
  max_class?: boolean;
}

export interface PaxMonTripInfo {
  tsi: TripServiceInfo;
  has_paxmon_data: boolean;
  all_edges_have_capacity_info: boolean;
  has_passengers: boolean;
}

export interface PaxMonCdfEntry {
  passengers: number;
  probability: number;
}

export const enum PaxMonCapacityType {
  Known,
  Unknown,
  Unlimited,
}

export interface PaxMonEdgeLoadInfo {
  from: Station;
  to: Station;
  departure_schedule_time: number;
  departure_current_time: number;
  arrival_schedule_time: number;
  arrival_current_time: number;
  capacity_type: PaxMonCapacityType;
  capacity: number;
  passenger_cdf: PaxMonCdfEntry[];
  updated: boolean;
  possibly_over_capacity: boolean;
  expected_passengers: number;

  // added by front end
  p_load_gt_100?: number;
  q_20?: number;
  q_50?: number;
  q_80?: number;
  q_5?: number;
  q_95?: number;
  min_pax?: number;
  max_pax?: number;
}

export interface PaxMonTripLoadInfo {
  tsi: TripServiceInfo;
  edges: PaxMonEdgeLoadInfo[];
}

export interface PaxMonFindTripsResponse {
  trips: PaxMonTripInfo[];
}

export interface PaxMonGetTripLoadInfosResponse {
  load_infos: PaxMonTripLoadInfo[];
}

export interface PaxMonStatusResponse {
  system_time: number;

  // system state
  tracked_groups?: number;

  // last update
  last_update_affected_groups?: number;
  last_update_affected_passengers?: number;
  last_update_broken_groups?: number;
  last_update_broken_passengers?: number;
}
