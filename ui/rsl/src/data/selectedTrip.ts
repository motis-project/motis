import { atomWithHash } from "jotai/utils";

import { TripServiceInfo } from "@/api/protocol/motis";

export const selectedTripAtom = atomWithHash<TripServiceInfo | undefined>(
  "trip",
  undefined
);
