// generated file - do not modify - run update-protocol to update
import { TripId } from "../motis";
import {
  PaxMonCompactJourney,
  PaxMonGroup,
  PaxMonLocalization,
  PaxMonLocalizationType,
  PaxMonTrackedUpdates,
  PaxMonTripLoadInfo,
} from "./paxmon";
import { RISContentType } from "./ris";

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
  planned_long_distance_destinations: string[];
  recommended_trip: TripId;
}

// paxforecast/Measures.fbs
export interface RtUpdateMeasure {
  recipients: MeasureRecipients;
  time: number;
  type: RISContentType;
  content: string;
}

// paxforecast/Measures.fbs
export type Measure =
  | TripLoadInfoMeasure
  | TripRecommendationMeasure
  | RtUpdateMeasure;

export type MeasureType =
  | "TripLoadInfoMeasure"
  | "TripRecommendationMeasure"
  | "RtUpdateMeasure";

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
}

// paxforecast/PaxForecastApplyMeasuresResponse.fbs
export interface PaxForecastApplyMeasuresStatistics {
  measure_time_points: number;
  total_measures_applied: number;
  total_affected_groups: number;
  total_alternative_routings: number;
  total_alternatives_found: number;
  t_rt_updates: number;
  t_get_affected_groups: number;
  t_find_alternatives: number;
  t_add_alternatives_to_graph: number;
  t_behavior_simulation: number;
  t_update_groups: number;
  t_update_tracker: number;
}

// paxforecast/PaxForecastApplyMeasuresResponse.fbs
export interface PaxForecastApplyMeasuresResponse {
  stats: PaxForecastApplyMeasuresStatistics;
  updates: PaxMonTrackedUpdates;
}

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastAlternative {
  journey: PaxMonCompactJourney;
  probability: number;
}

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastGroup {
  group: PaxMonGroup;
  localization_type: PaxMonLocalizationType;
  localization: PaxMonLocalization;
  forecast_alternatives: PaxForecastAlternative[];
}

// paxforecast/PaxForecastUpdate.fbs
export interface PaxForecastUpdate {
  universe: number;
  system_time: number;
  groups: PaxForecastGroup[];
  trips: PaxMonTripLoadInfo[];
}
