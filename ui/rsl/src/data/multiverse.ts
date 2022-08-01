import { atom } from "jotai";

export interface UniverseInfo {
  id: number;
  schedule: number;
  ttl: number; // seconds
}

export const multiverseIdAtom = atom(0);

export const defaultUniverse: UniverseInfo = { id: 0, schedule: 0, ttl: 0 };

export const universesAtom = atom<UniverseInfo[]>([defaultUniverse]);

// current selected universe/schedule
export const universeAtom = atom(0);
export const scheduleAtom = atom(0);
