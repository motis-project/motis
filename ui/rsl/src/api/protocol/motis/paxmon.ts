// generated file - do not modify - run update-protocol to update
import {
  Interval,
  Station,
  TripId,
  TripServiceInfo,
} from "@/api/protocol/motis";

// paxmon/PaxMonAddGroupsRequest.fbs
export interface PaxMonAddGroupsRequest {
  universe: number;
  groups: PaxMonGroup[];
}

// paxmon/PaxMonAddGroupsResponse.fbs
export interface PaxMonAddGroupsResponse {
  ids: number[];
}

// paxmon/PaxMonCombinedGroups.fbs
export interface PaxMonCombinedGroupRoutes {
  group_routes: PaxMonGroupRouteBaseInfo[];
  dist: PaxMonDistribution;
}

// paxmon/PaxMonCombinedGroups.fbs
export interface PaxMonCombinedGroupRouteIds {
  group_routes: PaxMonGroupWithRouteId[];
  dist: PaxMonDistribution;
}

// paxmon/PaxMonCompactJourney.fbs
export type PaxMonTransferType = "NONE" | "SAME_STATION" | "FOOTPATH";

// paxmon/PaxMonCompactJourney.fbs
export interface PaxMonTransferInfo {
  type: PaxMonTransferType;
  duration: number;
}

// paxmon/PaxMonCompactJourney.fbs
export interface PaxMonCompactJourneyLeg {
  trip: TripServiceInfo;
  enter_station: Station;
  exit_station: Station;
  enter_time: number;
  exit_time: number;
  enter_transfer: PaxMonTransferInfo;
}

// paxmon/PaxMonCompactJourney.fbs
export interface PaxMonCompactJourney {
  legs: PaxMonCompactJourneyLeg[];
}

// paxmon/PaxMonDestroyUniverseRequest.fbs
export interface PaxMonDestroyUniverseRequest {
  universe: number;
}

// paxmon/PaxMonDistribution.fbs
export interface PaxMonPdfEntry {
  n: number;
  p: number;
}

// paxmon/PaxMonDistribution.fbs
export interface PaxMonCdfEntry {
  n: number;
  p: number;
}

// paxmon/PaxMonDistribution.fbs
export interface PaxMonDistribution {
  min: number;
  max: number;
  q5: number;
  q50: number;
  q95: number;
  pdf: PaxMonPdfEntry[];
}

// paxmon/PaxMonFilterTripsRequest.fbs
export type PaxMonFilterTripsSortOrder =
  | "MostCritical"
  | "FirstDeparture"
  | "ExpectedPax"
  | "TrainNr"
  | "MaxLoad"
  | "EarliestCritical"
  | "MaxPaxRange";

// paxmon/PaxMonFilterTripsRequest.fbs
export type PaxMonFilterTripsTimeFilter =
  | "NoFilter"
  | "DepartureTime"
  | "DepartureOrArrivalTime";

// paxmon/PaxMonFilterTripsRequest.fbs
export interface PaxMonFilterTripsRequest {
  universe: number;
  ignore_past_sections: boolean;
  include_load_threshold: number;
  critical_load_threshold: number; // default: 1
  crowded_load_threshold: number; // default: 0.8
  include_edges: boolean;
  sort_by: PaxMonFilterTripsSortOrder;
  max_results: number;
  skip_first: number;
  filter_by_time: PaxMonFilterTripsTimeFilter;
  filter_interval: Interval;
  filter_by_train_nr: boolean;
  filter_train_nrs: number[];
  filter_by_service_class: boolean;
  filter_service_classes: number[];
}

// paxmon/PaxMonFilterTripsResponse.fbs
export interface PaxMonFilteredTripInfo {
  tsi: TripServiceInfo;
  section_count: number;
  critical_sections: number;
  crowded_sections: number;
  max_excess_pax: number;
  cumulative_excess_pax: number;
  max_load: number;
  max_expected_pax: number;
  edges: PaxMonEdgeLoadInfo[];
}

// paxmon/PaxMonFilterTripsResponse.fbs
export interface PaxMonFilterTripsResponse {
  total_matching_trips: number;
  filtered_trips: number;
  remaining_trips: number;
  next_skip: number;
  total_critical_sections: number;
  trips: PaxMonFilteredTripInfo[];
}

// paxmon/PaxMonFindTripsRequest.fbs
export interface PaxMonFindTripsRequest {
  universe: number;
  train_nr: number;
  only_trips_with_paxmon_data: boolean;
  filter_class: boolean;
  max_class: number;
}

// paxmon/PaxMonFindTripsResponse.fbs
export interface PaxMonTripInfo {
  tsi: TripServiceInfo;
  has_paxmon_data: boolean;
  all_edges_have_capacity_info: boolean;
  has_passengers: boolean;
}

// paxmon/PaxMonFindTripsResponse.fbs
export interface PaxMonFindTripsResponse {
  trips: PaxMonTripInfo[];
}

// paxmon/PaxMonForkUniverseRequest.fbs
export interface PaxMonForkUniverseRequest {
  universe: number;
  fork_schedule: boolean;
  ttl: number;
}

// paxmon/PaxMonForkUniverseResponse.fbs
export interface PaxMonForkUniverseResponse {
  universe: number;
  schedule: number;
  ttl: number;
}

// paxmon/PaxMonGetAddressableGroupsRequest.fbs
export interface PaxMonGetAddressableGroupsRequest {
  universe: number;
  trip: TripId;
}

// paxmon/PaxMonGetAddressableGroupsResponse.fbs
export interface PaxMonAddressableGroupsByFeeder {
  trip: TripServiceInfo;
  arrival_station: Station;
  arrival_schedule_time: number;
  arrival_current_time: number;
  cgs: PaxMonCombinedGroupRouteIds;
}

// paxmon/PaxMonGetAddressableGroupsResponse.fbs
export interface PaxMonAddressableGroupsByEntry {
  entry_station: Station;
  departure_schedule_time: number;
  cgs: PaxMonCombinedGroupRouteIds;
  by_feeder: PaxMonAddressableGroupsByFeeder[];
  starting_here: PaxMonCombinedGroupRouteIds;
}

// paxmon/PaxMonGetAddressableGroupsResponse.fbs
export interface PaxMonAddressableGroupsByInterchange {
  future_interchange: Station;
  cgs: PaxMonCombinedGroupRouteIds;
  by_entry: PaxMonAddressableGroupsByEntry[];
}

// paxmon/PaxMonGetAddressableGroupsResponse.fbs
export interface PaxMonAddressableGroupsSection {
  from: Station;
  to: Station;
  departure_schedule_time: number;
  departure_current_time: number;
  arrival_schedule_time: number;
  arrival_current_time: number;
  by_future_interchange: PaxMonAddressableGroupsByInterchange[];
}

// paxmon/PaxMonGetAddressableGroupsResponse.fbs
export interface PaxMonGetAddressableGroupsResponse {
  sections: PaxMonAddressableGroupsSection[];
  group_routes: PaxMonGroupRouteBaseInfo[];
}

// paxmon/PaxMonGetGroupsInTripRequest.fbs
export type PaxMonGroupFilter = "All" | "Entering" | "Exiting";

// paxmon/PaxMonGetGroupsInTripRequest.fbs
export type PaxMonGroupByStation =
  | "None"
  | "First"
  | "Last"
  | "FirstLongDistance"
  | "LastLongDistance"
  | "EntryAndLast";

// paxmon/PaxMonGetGroupsInTripRequest.fbs
export interface PaxMonGetGroupsInTripRequest {
  universe: number;
  trip: TripId;
  filter: PaxMonGroupFilter;
  group_by_station: PaxMonGroupByStation;
  group_by_other_trip: boolean;
  include_group_infos: boolean;
}

// paxmon/PaxMonGetGroupsInTripResponse.fbs
export interface GroupedPassengerGroups {
  grouped_by_station: Station[];
  grouped_by_trip: TripServiceInfo[];
  entry_station: Station[];
  entry_time: number;
  info: PaxMonCombinedGroupRoutes;
}

// paxmon/PaxMonGetGroupsInTripResponse.fbs
export interface GroupsInTripSection {
  from: Station;
  to: Station;
  departure_schedule_time: number;
  departure_current_time: number;
  arrival_schedule_time: number;
  arrival_current_time: number;
  groups: GroupedPassengerGroups[];
}

// paxmon/PaxMonGetGroupsInTripResponse.fbs
export interface PaxMonGetGroupsInTripResponse {
  sections: GroupsInTripSection[];
}

// paxmon/PaxMonGetGroupsRequest.fbs
export interface PaxMonGetGroupsRequest {
  universe: number;
  ids: number[];
  sources: PaxMonDataSource[];
}

// paxmon/PaxMonGetGroupsResponse.fbs
export interface PaxMonGetGroupsResponse {
  groups: PaxMonGroup[];
}

// paxmon/PaxMonGetInterchangesRequest.fbs
export interface PaxMonGetInterchangesRequest {
  universe: number;
  station: string;
  start_time: number;
  end_time: number;
  max_count: number;
  include_meta_stations: boolean;
  include_group_infos: boolean;
}

// paxmon/PaxMonGetInterchangesResponse.fbs
export interface PaxMonTripStopInfo {
  schedule_time: number;
  current_time: number;
  trips: TripServiceInfo[];
  station: Station;
}

// paxmon/PaxMonGetInterchangesResponse.fbs
export interface PaxMonInterchangeInfo {
  arrival: PaxMonTripStopInfo[];
  departure: PaxMonTripStopInfo[];
  groups: PaxMonCombinedGroupRoutes;
  transfer_time: number;
}

// paxmon/PaxMonGetInterchangesResponse.fbs
export interface PaxMonGetInterchangesResponse {
  station: Station;
  interchanges: PaxMonInterchangeInfo[];
  max_count_reached: boolean;
}

// paxmon/PaxMonGetTripLoadInfosRequest.fbs
export interface PaxMonGetTripLoadInfosRequest {
  universe: number;
  trips: TripId[];
}

// paxmon/PaxMonGetTripLoadInfosResponse.fbs
export interface PaxMonGetTripLoadInfosResponse {
  load_infos: PaxMonTripLoadInfo[];
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonDataSource {
  primary_ref: number;
  secondary_ref: number;
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroupRoute {
  index: number;
  journey: PaxMonCompactJourney;
  probability: number;
  planned_arrival_time: number;
  estimated_delay: number;
  source_flags: number;
  planned: boolean;
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroup {
  id: number;
  source: PaxMonDataSource;
  passenger_count: number;
  routes: PaxMonGroupRoute[];
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroupWithRoute {
  group_id: number;
  source: PaxMonDataSource;
  passenger_count: number;
  route: PaxMonGroupRoute;
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroupWithRouteId {
  g: number;
  r: number;
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroupRouteBaseInfo {
  g: number;
  r: number;
  n: number;
  p: number;
}

// paxmon/PaxMonGroup.fbs
export interface PaxMonGroupRouteUpdateInfo {
  g: number;
  r: number;
  n: number;
  p: number;
  pp: number;
}

// paxmon/PaxMonKeepAliveRequest.fbs
export interface PaxMonKeepAliveRequest {
  multiverse_id: number;
  universes: number[];
}

// paxmon/PaxMonKeepAliveResponse.fbs
export interface PaxMonUniverseKeepAliveInfo {
  universe: number;
  schedule: number;
  expires_in: number;
}

// paxmon/PaxMonKeepAliveResponse.fbs
export interface PaxMonKeepAliveResponse {
  multiverse_id: number;
  alive: PaxMonUniverseKeepAliveInfo[];
  expired: number[];
}

// paxmon/PaxMonLocalization.fbs
export interface PaxMonAtStation {
  station: Station;
  schedule_arrival_time: number;
  current_arrival_time: number;
  first_station: boolean;
}

// paxmon/PaxMonLocalization.fbs
export interface PaxMonInTrip {
  trip: TripId;
  next_station: Station;
  schedule_arrival_time: number;
  current_arrival_time: number;
}

// paxmon/PaxMonLocalization.fbs
export type PaxMonLocalization = PaxMonAtStation | PaxMonInTrip;

export type PaxMonLocalizationType = "PaxMonAtStation" | "PaxMonInTrip";

// paxmon/PaxMonLocalization.fbs
export interface PaxMonLocalizationWrapper {
  localization_type: PaxMonLocalizationType;
  localization: PaxMonLocalization;
}

// paxmon/PaxMonRemoveGroupsRequest.fbs
export interface PaxMonRemoveGroupsRequest {
  universe: number;
  ids: number[];
}

// paxmon/PaxMonRerouteGroupsRequest.fbs
export type PaxMonRerouteReason =
  | "MANUAL"
  | "TRANSFER_BROKEN"
  | "MAJOR_DELAY_EXPECTED"
  | "REVERT_FORECAST"
  | "SIMULATION";

// paxmon/PaxMonRerouteGroupsRequest.fbs
export interface PaxMonRerouteGroup {
  group: number;
  old_route_index: number;
  new_routes: PaxMonGroupRoute[];
  reason: PaxMonRerouteReason;
}

// paxmon/PaxMonRerouteGroupsRequest.fbs
export interface PaxMonRerouteGroupsRequest {
  universe: number;
  reroutes: PaxMonRerouteGroup[];
}

// paxmon/PaxMonRerouteGroupsResponse.fbs
export interface PaxMonRerouteRouteInfo {
  i: number;
  p: number;
  pp: number;
}

// paxmon/PaxMonRerouteGroupsResponse.fbs
export interface PaxMonRerouteGroupResult {
  group: number;
  old_route_index: number;
  new_routes: PaxMonRerouteRouteInfo[];
}

// paxmon/PaxMonRerouteGroupsResponse.fbs
export interface PaxMonRerouteGroupsResponse {
  reroutes: PaxMonRerouteGroupResult[];
}

// paxmon/PaxMonStatusRequest.fbs
export interface PaxMonStatusRequest {
  universe: number;
}

// paxmon/PaxMonStatusResponse.fbs
export interface PaxMonStatusResponse {
  system_time: number;
  multiverse_id: number;
  active_groups: number;
  trip_count: number;
}

// paxmon/PaxMonTrackedUpdates.fbs
export interface PaxMonCriticalTripInfo {
  critical_sections: number;
  max_excess_pax: number;
  cumulative_excess_pax: number;
}

// paxmon/PaxMonTrackedUpdates.fbs
export interface PaxMonUpdatedTrip {
  tsi: TripServiceInfo;
  rerouted: boolean;
  newly_critical_sections: number;
  no_longer_critical_sections: number;
  max_pax_increase: number;
  max_pax_decrease: number;
  critical_info_before: PaxMonCriticalTripInfo;
  critical_info_after: PaxMonCriticalTripInfo;
  updated_group_routes: PaxMonGroupWithRouteId[];
  before_edges: PaxMonEdgeLoadInfo[];
  after_edges: PaxMonEdgeLoadInfo[];
}

// paxmon/PaxMonTrackedUpdates.fbs
export interface PaxMonTrackedUpdates {
  updated_group_route_count: number;
  updated_trip_count: number;
  updated_trips: PaxMonUpdatedTrip[];
  updated_group_routes: PaxMonGroupRouteUpdateInfo[];
}

// paxmon/PaxMonTripLoadInfo.fbs
export type PaxMonCapacityType = "Known" | "Unknown" | "Unlimited";

// paxmon/PaxMonTripLoadInfo.fbs
export interface PaxMonEdgeLoadInfo {
  from: Station;
  to: Station;
  departure_schedule_time: number;
  departure_current_time: number;
  arrival_schedule_time: number;
  arrival_current_time: number;
  capacity_type: PaxMonCapacityType;
  capacity: number;
  dist: PaxMonDistribution;
  updated: boolean;
  possibly_over_capacity: boolean;
  prob_over_capacity: number;
  expected_passengers: number;
}

// paxmon/PaxMonTripLoadInfo.fbs
export interface PaxMonTripLoadInfo {
  tsi: TripServiceInfo;
  edges: PaxMonEdgeLoadInfo[];
}

// paxmon/PaxMonUniverseDestroyed.fbs
export interface PaxMonUniverseDestroyed {
  universe: number;
}

// paxmon/PaxMonUniverseForked.fbs
export interface PaxMonUniverseForked {
  base_universe: number;
  new_universe: number;
  new_schedule: number;
  schedule_forked: boolean;
}

// paxmon/PaxMonUpdate.fbs
export type PaxMonEventType =
  | "NO_PROBLEM"
  | "TRANSFER_BROKEN"
  | "MAJOR_DELAY_EXPECTED";

// paxmon/PaxMonUpdate.fbs
export type PaxMonReachabilityStatus =
  | "OK"
  | "BROKEN_INITIAL_ENTRY"
  | "BROKEN_TRANSFER_EXIT"
  | "BROKEN_TRANSFER_ENTRY"
  | "BROKEN_FINAL_EXIT";

// paxmon/PaxMonUpdate.fbs
export interface PaxMonEvent {
  type: PaxMonEventType;
  group_route: PaxMonGroupWithRoute;
  localization_type: PaxMonLocalizationType;
  localization: PaxMonLocalization;
  reachability_status: PaxMonReachabilityStatus;
  expected_arrival_time: number;
}

// paxmon/PaxMonUpdate.fbs
export interface PaxMonUpdate {
  universe: number;
  events: PaxMonEvent[];
}
