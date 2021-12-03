import React, { useState } from "react";

import { TripId } from "../api/protocol/motis";
import { usePaxMonGroupsInTripQuery } from "../api/paxmon";
import { PaxMonEdgeLoadInfoWithStats } from "../data/loadInfo";
import {
  GroupsInTripSection,
  PaxMonGroupByStation,
  PaxMonGroupFilter,
} from "../api/protocol/motis/paxmon";

import CombinedGroup from "./CombinedGroup";
import { useAtom } from "jotai";
import { universeAtom } from "../data/simulation";

function isSameSection(
  sec: GroupsInTripSection,
  selected: PaxMonEdgeLoadInfoWithStats | undefined
) {
  return (
    selected != undefined &&
    sec.departure_schedule_time === selected.departure_schedule_time &&
    sec.arrival_schedule_time === selected.arrival_schedule_time &&
    sec.from.id === selected.from.id &&
    sec.to.id === selected.to.id
  );
}

const groupFilters: Array<{ filter: PaxMonGroupFilter; label: string }> = [
  { filter: "All", label: "Alle" },
  { filter: "Entering", label: "Nur Einsteiger" },
  { filter: "Exiting", label: "Nur Aussteiger" },
];

const groupByStationOptions: Array<{
  groupBy: PaxMonGroupByStation;
  label: string;
}> = [
  { groupBy: "Last", label: "Letzter Halt" },
  { groupBy: "LastLongDistance", label: "Letzter Fernverkehrshalt" },
  { groupBy: "First", label: "Erster Halt" },
  { groupBy: "FirstLongDistance", label: "Erster Fernverkehrshalt" },
];

type TripSectionDetailsProps = {
  tripId: TripId;
  selectedSection: PaxMonEdgeLoadInfoWithStats | undefined;
  onClose: () => void;
};

function TripSectionDetails({
  tripId,
  selectedSection,
  onClose,
}: TripSectionDetailsProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [groupFilter, setGroupFilter] = useState<PaxMonGroupFilter>("Entering");
  const [groupByStation, setGroupByStation] =
    useState<PaxMonGroupByStation>("LastLongDistance");
  const [groupByOtherTrip, setGroupByOtherTrip] = useState(true);

  const {
    data: groupsInTrip,
    isLoading,
    error,
  } = usePaxMonGroupsInTripQuery({
    universe,
    trip: tripId,
    filter: groupFilter,
    group_by_station: groupByStation,
    group_by_other_trip: groupByOtherTrip,
    include_group_infos: false,
  });

  const groupByDirection =
    groupByStation === "First" || groupByStation === "FirstLongDistance"
      ? "Origin"
      : "Destination";

  const content = isLoading ? (
    <div>Loading trip section data..</div>
  ) : error || !groupsInTrip ? (
    <div>
      Error loading trip section data:{" "}
      {error instanceof Error ? error.message : error}
    </div>
  ) : (
    <div>
      {groupsInTrip.sections
        .filter((sec) => isSameSection(sec, selectedSection))
        .map((sec, secIdx) => (
          <div key={secIdx} className="mb-6">
            <div className="text-xl">
              <span>{sec.from.name}</span> → <span>{sec.to.name}</span>
            </div>
            <div>
              {sec.groups.length} Gruppen (
              {sec.groups.reduce(
                (sum, g) => sum + g.info.min_passenger_count,
                0
              )}
              {" - "}
              {sec.groups.reduce(
                (sum, g) => sum + g.info.max_passenger_count,
                0
              )}{" "}
              Reisende)
            </div>
            <ul>
              {sec.groups.slice(0, 30).map((gg, idx) => (
                <li key={idx}>
                  <CombinedGroup
                    plannedTrip={tripId}
                    combinedGroup={gg}
                    startStation={sec.from}
                    earliestDeparture={sec.departure_current_time}
                    groupByDirection={groupByDirection}
                  />
                </li>
              ))}
            </ul>
          </div>
        ))}
    </div>
  );

  return (
    <div className="mx-auto max-w-5xl">
      <form onSubmit={(e) => e.preventDefault()}>
        <div className="mb-5 flex flex-col gap-2">
          <div className="flex gap-4">
            <span>Gruppen anzeigen:</span>
            {groupFilters.map(({ filter, label }) => (
              <label key={filter} className="inline-flex items-center gap-1">
                <input
                  type="radio"
                  name="group-filter"
                  value={filter}
                  checked={groupFilter == filter}
                  onChange={() => setGroupFilter(filter)}
                />
                {label}
              </label>
            ))}
          </div>
          <div className="flex gap-4">
            <span>Gruppen zusammenfassen:</span>
            {groupByStationOptions.map(({ groupBy, label }) => (
              <label key={groupBy} className="inline-flex items-center gap-1">
                <input
                  type="radio"
                  name="group-by-station"
                  value={groupBy}
                  checked={groupByStation == groupBy}
                  onChange={() => setGroupByStation(groupBy)}
                />
                {label}
              </label>
            ))}
            {groupFilter !== "All" ? (
              <label className="inline-flex items-center gap-1">
                <input
                  type="checkbox"
                  name="group-by-other-trip"
                  checked={groupByOtherTrip}
                  onChange={() => setGroupByOtherTrip((val) => !val)}
                />
                {groupFilter === "Entering" ? "Zubringer" : "Abbringer"}
              </label>
            ) : null}
          </div>
          <div className="flex gap-4">
            <button
              type="button"
              onClick={onClose}
              className="bg-db-red-500 px-3 py-1 rounded text-white text-sm hover:bg-db-red-600"
            >
              Gruppenanzeige schließen
            </button>
          </div>
        </div>
      </form>
      {content}
    </div>
  );
}

export default TripSectionDetails;
