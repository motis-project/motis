import React from "react";
import { useQuery } from "react-query";
import { Station, TripId } from "./api/protocol/motis";
import { GroupsWithDestination } from "./api/protocol/motis/paxmon";
import { sendPaxForecastAlternativesRequest } from "./api/paxforecast";
import { formatTime } from "./util/dateFormat";
import TripView from "./TripView";

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
  const alternatives = data?.alternatives.filter(
    (alt) =>
      alt.compact_journey.legs.find(
        (l) => JSON.stringify(l.trip.trip) === plannedTripId
      ) === undefined
  );

  const groupInfo = (
    <div>
      {props.combinedGroup.min_passenger_count}-
      {props.combinedGroup.max_passenger_count} Reisende Richtung{" "}
      {props.combinedGroup.destination.name}
    </div>
  );

  const alternativesInfo = alternatives ? (
    <div>
      {alternatives.length}/{data?.alternatives.length} Alternativverbindungen
      (ab {formatTime(props.earliestDeparture)}):
      <ul>
        {alternatives.map((alt, idx) => (
          <li key={idx} className="pl-4">
            Ankunft um {formatTime(alt.arrival_time)} mit {alt.transfers}{" "}
            Umstiegen:
            <span>
              {alt.compact_journey.legs.map((leg, legIdx) => (
                <TripView key={legIdx} tsi={leg.trip} format="Short" />
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
    <div
      className="mt-2"
      onClick={() => {
        console.log(props, data, alternatives);
      }}
    >
      {groupInfo}
      {alternativesInfo}
    </div>
  );
}

export default CombinedGroup;
