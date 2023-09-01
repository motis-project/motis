import { Station } from "@/api/protocol/motis";
import {
  PaxMonCompactJourney,
  PaxMonCompactJourneyLeg,
  PaxMonGroup,
} from "@/api/protocol/motis/paxmon";

export function groupHasActiveUnreachableRoutes(group: PaxMonGroup): boolean {
  return group.routes.some((route) => route.destination_unreachable);
}

export function getDestinationStation(group: PaxMonGroup): Station {
  const legs = group.routes[0].journey.legs;
  return legs[legs.length - 1].exit_station;
}

export function getTotalPaxCount(groups: PaxMonGroup[]): number {
  return groups.reduce((total, group) => total + group.passenger_count, 0);
}

export function getJourneyLegAfterTransfer(
  journey: PaxMonCompactJourney,
  transferArrivalStation: Station | undefined,
  transferDepartureStation: Station | undefined,
): PaxMonCompactJourneyLeg | null {
  for (let i = 0; i < journey.legs.length; i++) {
    const leg = journey.legs[i];
    if (leg.enter_station.id === transferDepartureStation?.id) {
      return leg;
    } else if (
      leg.exit_station.id === transferArrivalStation?.id &&
      i < journey.legs.length - 2
    ) {
      return journey.legs[i + 1];
    }
  }

  return null;
}
