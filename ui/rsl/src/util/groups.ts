import { Station } from "@/api/protocol/motis";
import { PaxMonGroup } from "@/api/protocol/motis/paxmon";

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
