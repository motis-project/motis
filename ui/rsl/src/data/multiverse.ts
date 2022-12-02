import { atom } from "jotai";
import { atomWithStorage } from "jotai/utils";

export interface UniverseInfo {
  id: number;
  schedule: number;
  ttl: number; // seconds
}

export const multiverseIdAtom = atomWithStorage("multiverseId", 0);

export const defaultUniverse: UniverseInfo = { id: 0, schedule: 0, ttl: 0 };

export const universesAtom = atomWithStorage<UniverseInfo[]>("universes", [
  defaultUniverse,
]);

// current selected universe/schedule
export const universeAtom = atom(0);
export const scheduleAtom = atom(0);
