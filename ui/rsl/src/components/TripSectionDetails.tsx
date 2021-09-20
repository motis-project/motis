import React from "react";

import { TripId } from "../api/protocol/motis";
import { usePaxMonGroupsInTripQuery } from "../api/paxmon";
import { PaxMonEdgeLoadInfoWithStats } from "../data/loadInfo";
import { GroupsInTripSection } from "../api/protocol/motis/paxmon";

import CombinedGroup from "./CombinedGroup";

function isSameSection(
  sec: GroupsInTripSection,
  selected: PaxMonEdgeLoadInfoWithStats | null
) {
  return (
    selected !== null &&
    sec.departure_schedule_time === selected.departure_schedule_time &&
    sec.arrival_schedule_time === selected.arrival_schedule_time &&
    sec.from.id === selected.from.id &&
    sec.to.id === selected.to.id
  );
}

type TripSectionDetailsProps = {
  tripId: TripId;
  selectedSection: PaxMonEdgeLoadInfoWithStats | null;
};

function TripSectionDetails({
  tripId,
  selectedSection,
}: TripSectionDetailsProps): JSX.Element {
  const {
    data: groupsInTrip,
    isLoading,
    error,
  } = usePaxMonGroupsInTripQuery(tripId);

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
      {groupsInTrip.sections
        .filter((sec) => isSameSection(sec, selectedSection))
        .map((sec, secIdx) => (
          <div key={secIdx} className="mb-6">
            <div className="font-bold">
              <span>{sec.from.name}</span> → <span>{sec.to.name}</span>
            </div>
            <div>
              {sec.groups.length} Gruppen, {sec.groups_by_destination.length}{" "}
              unterschiedliche Ziele
            </div>
            <ul>
              {sec.groups_by_destination.slice(0, 20).map((gbd) => (
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
