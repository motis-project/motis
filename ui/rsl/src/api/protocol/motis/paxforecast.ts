// GENERATED FILE - DO NOT MODIFY
// -> see /tools/protocol for information on how to update this file
import { TripId } from "@/api/protocol/motis";
import {
  PaxMonCompactJourney,
  PaxMonGroupWithRoute,
  PaxMonLocalization,
  PaxMonLocalizationType,
  PaxMonTrackedUpdates,
  PaxMonTripLoadInfo,
} from "@/api/protocol/motis/paxmon";
import { RISContentType } from "@/api/protocol/motis/ris";

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastAlternative {
  journey: PaxMonCompactJourney;
  probability: number;
}

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastGroupRoute {
  group_route: PaxMonGroupWithRoute;
  localization_type: PaxMonLocalizationType;
  localization: PaxMonLocalization;
  forecast_alternatives: PaxForecastAlternative[];
}

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastUpdate {
  universe: number;
  system_time: number;
  groups: PaxForecastGroupRoute[];
  trips: PaxMonTripLoadInfo[];
}

// paxforecast/Measures.fbs
export interface MeasureRecipients {
  trips: TripId[];
  stations: string[];
}

// paxforecast/Measures.fbs
export type LoadLevel = "Unknown" | "Low" | "NoSeats" | "Full";

// paxforecast/Measures.fbs
export interface TripLoadInfoMeasure {
  recipients: MeasureRecipients;
  time: number;
  trip: TripId;
  level: LoadLevel;
}

// paxforecast/Measures.fbs
export interface TripRecommendationMeasure {
  recipients: MeasureRecipients;
  time: number;
  planned_trips: TripId[];
  planned_destinations: string[];
  recommended_trip: TripId;
}

// paxforecast/Measures.fbs
export interface TripWithLoadLevel {
  trip: TripId;
  level: LoadLevel;
}

// paxforecast/Measures.fbs
export interface TripLoadRecommendationMeasure {
  recipients: MeasureRecipients;
  time: number;
  planned_destinations: string[];
  full_trips: TripWithLoadLevel[];
  recommended_trips: TripWithLoadLevel[];
}

// paxforecast/Measures.fbs
export interface RtUpdateMeasure {
  recipients: MeasureRecipients;
  time: number;
  type: RISContentType;
  content: string;
}

// paxforecast/Measures.fbs
export interface UpdateCapacitiesMeasure {
  time: number;
  file_contents: string[];
  remove_existing_trip_capacities: boolean;
  remove_existing_category_capacities: boolean;
  remove_existing_vehicle_capacities: boolean;
  remove_existing_trip_formations: boolean;
  remove_existing_gattung_capacities: boolean;
  remove_existing_baureihe_capacities: boolean;
  remove_existing_vehicle_group_capacities: boolean;
  remove_existing_overrides: boolean;
  track_trip_updates: boolean;
}

// paxforecast/Measures.fbs
export interface OverrideCapacitySection {
  departure_station: string;
  departure_schedule_time: number;
  seats: number;
}

// paxforecast/Measures.fbs
export interface OverrideCapacityMeasure {
  time: number;
  trip: TripId;
  sections: OverrideCapacitySection[];
}

// paxforecast/Measures.fbs
export type Measure =
  | TripLoadInfoMeasure
  | TripRecommendationMeasure
  | TripLoadRecommendationMeasure
  | RtUpdateMeasure
  | UpdateCapacitiesMeasure
  | OverrideCapacityMeasure;

export type MeasureType =
  | "TripLoadInfoMeasure"
  | "TripRecommendationMeasure"
  | "TripLoadRecommendationMeasure"
  | "RtUpdateMeasure"
  | "UpdateCapacitiesMeasure"
  | "OverrideCapacityMeasure";

// paxforecast/Measures.fbs
export interface MeasureWrapper {
  measure_type: MeasureType;
  measure: Measure;
}

// paxforecast/PaxForecastApplyMeasuresRequest.fbs
export interface PaxForecastApplyMeasuresRequest {
  universe: number;
  measures: MeasureWrapper[];
  replace_existing: boolean;
  preparation_time: number;
  include_before_trip_load_info: boolean;
  include_after_trip_load_info: boolean;
  include_trips_with_unchanged_load: boolean;
}

// paxforecast/PaxForecastApplyMeasuresResponse.fbs
export interface PaxForecastApplyMeasuresStatistics {
  measure_time_points: number;
  total_measures_applied: number;
  total_affected_groups: number;
  total_alternative_routings: number;
  total_alternatives_found: number;
  group_routes_broken: number;
  group_routes_with_major_delay: number;
  t_rt_updates: number;
  t_get_affected_groups: number;
  t_find_alternatives: number;
  t_add_alternatives_to_graph: number;
  t_behavior_simulation: number;
  t_update_groups: number;
  t_update_tracker: number;
  t_update_capacities: number;
}

// paxforecast/PaxForecastApplyMeasuresResponse.fbs
export interface PaxForecastApplyMeasuresResponse {
  stats: PaxForecastApplyMeasuresStatistics;
  updates: PaxMonTrackedUpdates;
}

// paxforecast/PaxForecastMetricsRequest.fbs
export interface PaxForecastMetricsRequest {
  universe: number;
}

// paxforecast/PaxForecastMetricsResponse.fbs
export interface PaxForecastMetrics {
  start_time: number;
  entries: number;
  monitoring_events: number[];
  group_routes: number[];
  major_delay_group_routes: number[];
  routing_requests: number[];
  alternatives_found: number[];
  rerouted_group_routes: number[];
  removed_group_routes: number[];
  major_delay_group_routes_with_alternatives: number[];
  total_timing: number[];
}

// paxforecast/PaxForecastMetricsResponse.fbs
export interface PaxForecastMetricsResponse {
  by_system_time: PaxForecastMetrics;
  by_processing_time: PaxForecastMetrics;
}
