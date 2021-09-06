import React from "react";
import { useQuery } from "react-query";

import { Station, TripId } from "./api/protocol/motis";
import { GroupsInTripSection } from "./api/protocol/motis/paxmon";
import { sendPaxMonGroupsInTripRequest } from "./api/paxmon";
import { sendPaxForecastAlternativesRequest } from "./api/paxforecast";

type TripSectionDetailsProps = {
  tripId: TripId;
};

async function requestAlternatives(
  sec: GroupsInTripSection,
  destination: Station
) {
  const res = await sendPaxForecastAlternativesRequest({
    start_type: "PaxMonAtStation",
    start: {
      station: sec.from,
      current_arrival_time: sec.arrival_current_time,
      schedule_arrival_time: sec.departure_schedule_time,
      first_station: false,
    },
    destination,
  });
  console.log(res);
}

function TripSectionDetails({ tripId }: TripSectionDetailsProps): JSX.Element {
  const {
    data: groupsInTrip,
    isLoading,
    error,
  } = useQuery(["trip", "groups", { tripId }], () =>
    sendPaxMonGroupsInTripRequest(tripId)
  );

  if (isLoading) {
    return <div>Loading trip section data..</div>;
  } else if (error || !groupsInTrip) {
    return <div>Error loading trip section data: {error}</div>;
  }

  return (
    <div className="mx-auto max-w-5xl">
      {groupsInTrip.sections.map((sec, secIdx) => (
        <div key={secIdx} className="mb-3">
          <div>
            <span>{sec.from.name}</span> → <span>{sec.to.name}</span>
          </div>
          <div>
            {sec.groups.length} Gruppen, {sec.groups_by_destination.length}{" "}
            unterschiedliche Ziele
          </div>
          <ul>
            {sec.groups_by_destination.slice(0, 5).map((gbd) => (
              <li
                key={gbd.destination.id}
                className="cursor-pointer"
                onClick={() => requestAlternatives(sec, gbd.destination)}
              >
                {gbd.min_passenger_count}-{gbd.max_passenger_count} Reisende
                Richtung {gbd.destination.name}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default TripSectionDetails;
