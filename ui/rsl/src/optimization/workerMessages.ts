import { TripId } from "@/api/protocol/motis";

import { MeasureUnion } from "@/data/measures";

export type OptimizationType = "V1";

// sent to worker
export type WorkerRequest =
  | { action: "Init"; apiEndpoint: string }
  | {
      action: "Start";
      universe: number;
      schedule: number;
      tripId: TripId;
      optType: OptimizationType;
    };

// sent from worker
export type WorkerUpdate =
  | { type: "Log"; msg: string }
  | { type: "UniverseForked"; universe: number }
  | { type: "UniverseDestroyed"; universe: number }
  | { type: "MeasuresAdded"; measures: MeasureUnion[] }
  | { type: "OptimizationComplete" };
