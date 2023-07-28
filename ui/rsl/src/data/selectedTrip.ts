import { atom } from "jotai";

import { TripServiceInfo } from "@/api/protocol/motis";

export const mostRecentlySelectedTripAtom = atom<TripServiceInfo | undefined>(
  undefined,
);
