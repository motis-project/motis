import { PrimitiveAtom, atom } from "jotai";

import {
  MeasureWrapper,
  PaxForecastApplyMeasuresResponse,
} from "@/api/protocol/motis/paxforecast";

export interface SimulationResult {
  universe: number;
  startedAt: Date;
  finishedAt: Date;
  measures: MeasureWrapper[];
  response: PaxForecastApplyMeasuresResponse;
}

export const simResultsAtom = atom<PrimitiveAtom<SimulationResult>[]>([]);

export const hasSimResultsAtom = atom(
  (get) => get(simResultsAtom).length !== 0,
);

export const selectedSimResultAtom =
  atom<PrimitiveAtom<SimulationResult> | null>(null);
