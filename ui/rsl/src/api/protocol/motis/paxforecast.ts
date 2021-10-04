// generated file - do not modify - run update-protocol to update

import { TripId } from "../motis";
import {
  PaxMonCompactJourney,
  PaxMonGroup,
  PaxMonLocalization,
  PaxMonLocalizationType,
  PaxMonTripLoadInfo,
} from "./paxmon";

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
  interchange_station: string;
}

// paxforecast/Measures.fbs
export type Measure = TripLoadInfoMeasure | TripRecommendationMeasure;

export type MeasureType = "TripLoadInfoMeasure" | "TripRecommendationMeasure";

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
