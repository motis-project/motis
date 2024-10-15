import * as Tooltip from "@radix-ui/react-tooltip";
import { useAtom } from "jotai";
import { useQuery } from "react-query";

import { Station, TripId } from "@/api/protocol/motis";
import { GroupedPassengerGroups } from "@/api/protocol/motis/paxmon";

import { sendRoutingRequest } from "@/api/routing";

import {
  Journey,
  connectionToJourney,
  getArrivalTime,
  getDepartureTime,
} from "@/data/journey";
import { scheduleAtom } from "@/data/simulation";

import { formatTime } from "@/util/dateFormat";

import JourneyTripNameView from "@/components/JourneyTripNameView";
import { MiniTripLoadGraphForTrip } from "@/components/MiniTripLoadGraph";

export type GroupByDirection = "Origin" | "Destination" | "None";

export type CombinedGroupProps = {
  plannedTrip: TripId;
  combinedGroup: GroupedPassengerGroups;
  startStation: Station;
  earliestDeparture: number;
  groupByDirection: GroupByDirection;
};

const SEARCH_INTERVAL = 61;

function CombinedGroup({
  plannedTrip,
  combinedGroup,
  startStation,
  earliestDeparture,
  groupByDirection,
}: CombinedGroupProps): JSX.Element {
  const destinationStation = combinedGroup.grouped_by_station[0];
  const previousTrip = combinedGroup.grouped_by_trip[0];
  const findAlternatives = groupByDirection === "Destination";

  const [schedule] = useAtom(scheduleAtom);

  const { data, isLoading, error } = useQuery(
    [
      "alternatives",
      {
        from: startStation.id,
        to: destinationStation?.id,
        earliestDeparture: earliestDeparture,
        intervalDuration: SEARCH_INTERVAL,
      },
    ],
    () =>
      sendRoutingRequest({
        start_type: "PretripStart",
        start: {
          station: { id: startStation.id, name: "" },
          interval: {
            begin: earliestDeparture,
            end: earliestDeparture + SEARCH_INTERVAL * 60,
          },
          min_connection_count: 0,
          extend_interval_earlier: false,
          extend_interval_later: false,
        },
        destination: { id: destinationStation.id, name: "" },
        search_type: "Default",
        search_dir: "Forward",
        via: [],
        additional_edges: [],
        use_start_metas: true,
        use_dest_metas: true,
        use_start_footpaths: true,
        schedule,
      }),
    { enabled: findAlternatives }
  );

  const plannedTripId = JSON.stringify(plannedTrip);
  const containsCurrentTrip = (j: Journey) =>
    j.tripLegs.find((leg) =>
      leg.trips.find((t) => JSON.stringify(t.trip.id) === plannedTripId)
    ) !== undefined;

  // TODO: group in 2 levels: destination, origin trip (?)

  const groupInfo = (
    <div>
      <span className="font-bold">
        {combinedGroup.info.dist.q5 == combinedGroup.info.dist.q95
          ? `${combinedGroup.info.dist.q5} Reisende`
          : `${combinedGroup.info.dist.q5} - ${combinedGroup.info.dist.q95} Reisende`}
        {groupByDirection !== "None" && (
          <>
            {groupByDirection === "Origin"
              ? " mit Reisebeginn in "
              : " mit Ziel "}
            {destinationStation.name}
          </>
        )}
        {combinedGroup.entry_station.length === 1
          ? `, Einstieg in ${combinedGroup.entry_station[0].name}`
          : null}
        {previousTrip &&
          ` und Ankunft mit ${
            previousTrip.service_infos[0]?.category ?? "Zug"
          } ${
            previousTrip.service_infos[0]?.train_nr ??
            previousTrip.trip.train_nr
          }`}
      </span>
    </div>
  );

  const journeys = data?.connections?.map((c) => connectionToJourney(c));

  const alternativesInfo = journeys ? (
    <div>
      {`${journeys.length} Mögliche Alternative(n) (ab ${formatTime(
        earliestDeparture
      )}):`}
      <ul>
        {journeys.map((j, idx) => (
          <li
            key={idx}
            className={`pl-4 ${containsCurrentTrip(j) ? "text-gray-400" : ""}`}
          >
            {formatTime(getDepartureTime(j))} &rarr;{" "}
            {formatTime(getArrivalTime(j))}, {j.transfers} Umstiege:
            <span className="inline-flex gap-3 pl-2">
              {j.tripLegs.map((leg, legIdx) => (
                <Tooltip.Root key={legIdx}>
                  <Tooltip.Trigger className="cursor-default">
                    <JourneyTripNameView jt={leg.trips[0]} />
                  </Tooltip.Trigger>
                  <Tooltip.Content>
                    <div className="w-96 bg-white p-2 rounded-md shadow-lg flex justify-center">
                      <MiniTripLoadGraphForTrip tripId={leg.trips[0].trip.id} />
                    </div>
                    <Tooltip.Arrow className="text-white fill-current" />
                  </Tooltip.Content>
                </Tooltip.Root>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </div>
  ) : isLoading ? (
    <div>Suche nach Alternativverbindungen...</div>
  ) : (
    <div>Fehler: {error instanceof Error ? error.message : error}</div>
  );

  return (
    <div className="mt-8">
      {groupInfo}
      {findAlternatives ? alternativesInfo : null}
    </div>
  );
}

export default CombinedGroup;
