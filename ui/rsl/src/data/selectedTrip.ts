import { atom } from "jotai";

import { TripServiceInfo } from "@/api/protocol/motis";

export const selectedTripAtom = atom<TripServiceInfo | undefined>(undefined);
