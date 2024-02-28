import { Station, TripServiceInfo } from "@/api/protocol/motis";

// sent to worker
export interface WorkerRequest {
  action: "Start";
  apiEndpoint: string;
}

// sent from worker
export type WorkerUpdate =
  | { type: "TripCount"; totalTrips: number }
  | {
      type: "TripInfo";
      progress: number;
      result: TripEvalResult;
    }
  | {
      type: "Done";
      progress: number;
      totalTrips: number;
      evaluatedTrips: number;
      result: EvalResult;
    };

export interface TripEvalResult {
  tsi: TripServiceInfo;
  totalSectionCount: number;
  evaluatedSectionCount: number;

  sections: TripEvalSectionInfo[];

  // rsl
  deviationAvg: number;

  // comparison
  q50Mae: number;
  q50Mse: number;
  expectedMae: number;
  expectedMse: number;
}

export interface TripEvalSectionInfo {
  from: Station;
  to: Station;
  departureScheduleTime: number;
  departureCurrentTime: number;
  arrivalScheduleTime: number;
  arrivalCurrentTime: number;
  duration: number; // minutes

  // check data
  checkCount: number;
  checkPaxMin: number;
  checkPaxMax: number;
  checkPaxAvg: number;
  checkSpread: number;

  // rsl
  expectedPax: number;
  forecastPaxQ5: number;
  forecastPaxQ50: number;
  forecastPaxQ95: number;
  forecastSpread: number;
  deviation: number;

  // comparison
  q50Diff: number;
  q50Factor: number;
  expectedDiff: number;
  expectedFactor: number;
}

export interface EvalResult {
  trips: TripEvalResult[];

  intervalStart: number;
  intervalEnd: number;

  q50Mae: number;
  q50Mse: number;

  tripCsv: string;
  sectionCsv: string;
}
