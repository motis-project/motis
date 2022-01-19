import { useQuery } from "react-query";
import { useAtom } from "jotai";
import * as Tooltip from "@radix-ui/react-tooltip";

import { Station, TripId } from "../api/protocol/motis";
import { GroupedPassengerGroups } from "../api/protocol/motis/paxmon";
import { sendRoutingRequest } from "../api/routing";
import { connectionToJourney, Journey } from "../data/journey";
import { formatTime } from "../util/dateFormat";
import { scheduleAtom } from "../data/simulation";

import TripLoadForecastChart from "./TripLoadForecastChart";
import JourneyTripNameView from "./JourneyTripNameView";
import TripServiceInfoView from "./TripServiceInfoView";

export type GroupByDirection = "Origin" | "Destination";

export type CombinedGroupProps = {
  plannedTrip: TripId;
  combinedGroup: GroupedPassengerGroups;
  startStation: Station;
  earliestDeparture: number;
  groupByDirection: GroupByDirection;
};

const SEARCH_INTERVAL = 61;

function getArrivalTime(j: Journey): number {
  const finalLeg = j.legs[j.legs.length - 1];
  switch (finalLeg.type) {
    case "trip":
      return finalLeg.stops[finalLeg.stops.length - 1].arrival.time;
    case "walk":
      return finalLeg.to.arrival.time;
  }
}

function getDepartureTime(j: Journey): number {
  const firstLeg = j.legs[0];
  switch (firstLeg.type) {
    case "trip":
      return firstLeg.stops[0].departure.time;
    case "walk":
      return firstLeg.from.departure.time;
  }
}

function CombinedGroup(props: CombinedGroupProps): JSX.Element {
  const destinationStation = props.combinedGroup.grouped_by_station[0];
  const previousTrip = props.combinedGroup.grouped_by_trip[0];
  const findAlternatives = props.groupByDirection === "Destination";

  const [schedule] = useAtom(scheduleAtom);

  const { data, isLoading, error } = useQuery(
    [
      "alternatives",
      {
        from: props.startStation.id,
        to: destinationStation.id,
        earliestDeparture: props.earliestDeparture,
        intervalDuration: SEARCH_INTERVAL,
      },
    ],
    () =>
      sendRoutingRequest({
        start_type: "PretripStart",
        start: {
          station: { id: props.startStation.id, name: "" },
          interval: {
            begin: props.earliestDeparture,
            end: props.earliestDeparture + SEARCH_INTERVAL * 60,
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

  const plannedTripId = JSON.stringify(props.plannedTrip);
  const containsCurrentTrip = (j: Journey) =>
    j.tripLegs.find((leg) =>
      leg.trips.find((t) => JSON.stringify(t.trip.id) === plannedTripId)
    ) !== undefined;

  // TODO: group in 2 levels: destination, origin trip (?)

  const groupInfo = (
    <div>
      <span className="font-bold">
        {props.combinedGroup.info.min_passenger_count}-
        {props.combinedGroup.info.max_passenger_count} Reisende
        {props.groupByDirection === "Origin" ? " aus " : " in "}Richtung{" "}
        {destinationStation.name}
      </span>
      {previousTrip && (
        <div>
          Ankunft mit: <TripServiceInfoView tsi={previousTrip} format="Long" />
        </div>
      )}
    </div>
  );

  const journeys = data?.connections?.map((c) => connectionToJourney(c));

  const alternativesInfo = journeys ? (
    <div>
      {journeys.length} Mögliche Verbindungen (ab{" "}
      {formatTime(props.earliestDeparture)}):
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
                      <TripLoadForecastChart
                        tripId={leg.trips[0].trip.id}
                        mode="Tooltip"
                      />
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
