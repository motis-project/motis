import React from "react";
import { useQuery } from "react-query";
import * as Tooltip from "@radix-ui/react-tooltip";

import { Station, TripId } from "../api/protocol/motis";
import { GroupsWithDestination } from "../api/protocol/motis/paxmon";
import { sendPaxForecastAlternativesRequest } from "../api/paxforecast";
import { formatTime } from "../util/dateFormat";
import { Alternative } from "../api/protocol/motis/paxforecast";

import TripView from "./TripView";
import TripLoadForecastChart from "./TripLoadForecastChart";

type CombinedGroupProps = {
  plannedTrip: TripId;
  combinedGroup: GroupsWithDestination;
  startStation: Station;
  earliestDeparture: number;
};

function CombinedGroup(props: CombinedGroupProps): JSX.Element {
  const { data, isLoading, error } = useQuery(
    [
      "alternatives",
      {
        from: props.startStation.id,
        to: props.combinedGroup.destination.id,
        earliestDeparture: props.earliestDeparture,
      },
    ],
    () =>
      sendPaxForecastAlternativesRequest({
        start_type: "PaxMonAtStation",
        start: {
          station: props.startStation,
          current_arrival_time: props.earliestDeparture,
          schedule_arrival_time: props.earliestDeparture,
          first_station: true,
        },
        destination: props.combinedGroup.destination,
        interval_duration: 60,
      })
  );

  const plannedTripId = JSON.stringify(props.plannedTrip);
  const containsCurrentTrip = (alt: Alternative) =>
    alt.compact_journey.legs.find(
      (l) => JSON.stringify(l.trip.trip) === plannedTripId
    ) !== undefined;

  const groupInfo = (
    <div>
      {props.combinedGroup.min_passenger_count}-
      {props.combinedGroup.max_passenger_count} Reisende Richtung{" "}
      {props.combinedGroup.destination.name}
    </div>
  );

  const alternativesInfo = data?.alternatives ? (
    <div>
      {data.alternatives.length} Mögliche Verbindungen (ab{" "}
      {formatTime(props.earliestDeparture)}):
      <ul>
        {data.alternatives.map((alt, idx) => (
          <li
            key={idx}
            className={`pl-4 ${
              containsCurrentTrip(alt) ? "text-gray-400" : ""
            }`}
          >
            Ankunft um {formatTime(alt.arrival_time)} mit {alt.transfers}{" "}
            Umstiegen:
            <span className="inline-flex gap-3 pl-2">
              {alt.compact_journey.legs.map((leg, legIdx) => (
                <Tooltip.Root key={legIdx}>
                  <Tooltip.Trigger>
                    <TripView tsi={leg.trip} format="Short" />
                  </Tooltip.Trigger>
                  <Tooltip.Content>
                    <div className="w-96 bg-white p-2 rounded-md shadow-lg flex justify-center">
                      <TripLoadForecastChart
                        tripId={leg.trip.trip}
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
    <div className="mt-2">
      {groupInfo}
      {alternativesInfo}
    </div>
  );
}

export default CombinedGroup;
