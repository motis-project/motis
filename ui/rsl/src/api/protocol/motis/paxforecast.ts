// generated file - do not modify - run update-protocol to update

import { TripId, Interval, Station } from "../motis";
import {
  PaxMonLocalization,
  PaxMonLocalizationType,
  PaxMonCompactJourney,
  PaxMonGroup,
  PaxMonTripLoadInfo,
} from "./paxmon";

// paxforecast/Measures.fbs
export interface MeasureRecipients {
  trips: TripId[];
  stations: string[];
}

// paxforecast/Measures.fbs
export type LoadLevel = "Low" | "NoSeats" | "Full";

// paxforecast/Measures.fbs
export interface TripLoadInfoMeasure {
  recipients: MeasureRecipients;
  interval: Interval;
  trip: TripId;
  level: LoadLevel;
}

// paxforecast/Measures.fbs
export interface TripRecommendationMeasure {
  recipients: MeasureRecipients;
  interval: Interval;
  planned_trips: TripId[];
  planned_destinations: string[];
  recommended_trip: TripId;
}

// paxforecast/Measures.fbs
export type Measure = TripLoadInfoMeasure | TripRecommendationMeasure;

export type MeasureType = "TripLoadInfoMeasure" | "TripRecommendationMeasure";

// paxforecast/Measures.fbs
export interface MeasureWrapper {
  measure_type: MeasureType;
  measure: Measure;
}

// paxforecast/PaxForecastAlternativesRequest.fbs
export interface PaxForecastAlternativesRequest {
  start_type: PaxMonLocalizationType;
  start: PaxMonLocalization;
  destination: Station;
  interval_duration: number;
}

// paxforecast/PaxForecastAlternativesResponse.fbs
export interface Alternative {
  compact_journey: PaxMonCompactJourney;
  arrival_time: number;
  duration: number;
  transfers: number;
}

// paxforecast/PaxForecastAlternativesResponse.fbs
export interface PaxForecastAlternativesResponse {
  alternatives: Alternative[];
}

// paxforecast/PaxForecastApplyMeasuresRequest.fbs
export interface PaxForecastApplyMeasuresRequest {
  measures: MeasureWrapper[];
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
  system_time: number;
  groups: PaxForecastGroup[];
  trips: PaxMonTripLoadInfo[];
}
