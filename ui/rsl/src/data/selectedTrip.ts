import { atom } from "jotai";

import { TripId } from "@/api/protocol/motis";

export const selectedTripAtom = atom<TripId | undefined>(undefined);
