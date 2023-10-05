import { Station, TripId } from "@/api/protocol/motis";
import {
  PaxMonAtStation,
  PaxMonInTrip,
  PaxMonLocalizationWrapper,
} from "@/api/protocol/motis/paxmon";

export function getNextStation(loc: PaxMonLocalizationWrapper): Station {
  switch (loc.localization_type) {
    case "PaxMonAtStation":
      return (loc.localization as PaxMonAtStation).station;
    case "PaxMonInTrip":
      return (loc.localization as PaxMonInTrip).next_station;
  }
}

export function getCurrentTripId(
  loc: PaxMonLocalizationWrapper,
): TripId | null {
  switch (loc.localization_type) {
    case "PaxMonAtStation":
      return null;
    case "PaxMonInTrip":
      return (loc.localization as PaxMonInTrip).trip;
  }
}

export function canSwitchLocalization(
  from: PaxMonLocalizationWrapper,
  to: PaxMonLocalizationWrapper,
): boolean {
  return (
    getNextStation(from).id === getNextStation(to).id &&
    JSON.stringify(getCurrentTripId(from)) ===
      JSON.stringify(getCurrentTripId(to))
  );
}
