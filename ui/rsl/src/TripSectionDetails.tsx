import React from "react";
import { useQuery } from "react-query";

import { TripId } from "./api/protocol/motis";
import { sendPaxMonGroupsInTripRequest } from "./api/paxmon";
import CombinedGroup from "./CombinedGroup";

type TripSectionDetailsProps = {
  tripId: TripId;
};

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
    return (
      <div>
        Error loading trip section data:{" "}
        {error instanceof Error ? error.message : error}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl">
      {groupsInTrip.sections.map((sec, secIdx) => (
        <div key={secIdx} className="mb-6">
          <div>
            <span>{sec.from.name}</span> → <span>{sec.to.name}</span>
          </div>
          <div>
            {sec.groups.length} Gruppen, {sec.groups_by_destination.length}{" "}
            unterschiedliche Ziele
          </div>
          <ul>
            {sec.groups_by_destination.slice(0, 5).map((gbd) => (
              <li key={gbd.destination.id}>
                <CombinedGroup
                  plannedTrip={tripId}
                  combinedGroup={gbd}
                  startStation={sec.from}
                  earliestDeparture={sec.departure_current_time}
                />
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default TripSectionDetails;
