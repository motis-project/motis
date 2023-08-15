import { useAtom } from "jotai";
import { useState } from "react";

import { Station, TripId } from "@/api/protocol/motis";
import {
  GroupedPassengerGroups,
  GroupsInTripSection,
  PaxMonEdgeLoadInfo,
  PaxMonGroupByStation,
  PaxMonGroupFilter,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonGroupsInTripQuery } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import CombinedGroup from "@/components/trips/CombinedGroup";

function isSameSection(
  sec: GroupsInTripSection,
  selected: PaxMonEdgeLoadInfo | undefined,
) {
  return (
    selected != undefined &&
    sec.departure_schedule_time === selected.departure_schedule_time &&
    sec.arrival_schedule_time === selected.arrival_schedule_time &&
    sec.from.id === selected.from.id &&
    sec.to.id === selected.to.id
  );
}

const groupFilters: { filter: PaxMonGroupFilter; label: string }[] = [
  { filter: "All", label: "Alle" },
  { filter: "Entering", label: "Nur Einsteiger" },
  { filter: "Exiting", label: "Nur Aussteiger" },
];

const groupByStationOptions: {
  groupBy: PaxMonGroupByStation;
  label: string;
}[] = [
  { groupBy: "None", label: "Keine" },
  { groupBy: "Last", label: "Letzter Halt" },
  //{ groupBy: "LastLongDistance", label: "Letzter FV-Halt" },
  { groupBy: "First", label: "Erster Halt" },
  //{ groupBy: "FirstLongDistance", label: "Erster FV-Halt" },
  { groupBy: "EntryAndLast", label: "Einstiegshalt und Ziel" },
];

interface TripSectionDetailsProps {
  tripId: TripId;
  selectedSection: PaxMonEdgeLoadInfo | undefined;
}

function TripSectionDetails({
  tripId,
  selectedSection,
}: TripSectionDetailsProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [groupFilter, setGroupFilter] = useState<PaxMonGroupFilter>("All");
  const [groupByStation, setGroupByStation] =
    useState<PaxMonGroupByStation>("EntryAndLast");
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
    include_group_infos: true,
  });

  const groupByDirection =
    groupByStation === "First" || groupByStation === "FirstLongDistance"
      ? "Origin"
      : groupByStation === "None"
      ? "None"
      : "Destination";

  const getMinPaxInSection = (sec: GroupsInTripSection) =>
    sec.groups.reduce((sum, g) => sum + g.info.dist.q5, 0);
  const getMaxPaxInSection = (sec: GroupsInTripSection) =>
    sec.groups.reduce((sum, g) => sum + g.info.dist.q95, 0);

  const content = isLoading ? (
    <div>Loading trip section data..</div>
  ) : error || !groupsInTrip ? (
    <div>
      Error loading trip section data:{" "}
      {error instanceof Error ? error.message : `Unbekannter Fehler`}
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
              {`${sec.groups.length} Gruppen (${getMinPaxInSection(
                sec,
              )} - ${getMaxPaxInSection(sec)} Reisende)`}
              {groupFilter == "All" && " in dem Fahrtabschnitt"}
              {groupFilter == "Entering" && ` mit Einstieg in ${sec.from.name}`}
              {groupFilter == "Exiting" && ` mit Ausstieg in ${sec.to.name}`}
            </div>
            <ul>
              {sec.groups
                .filter((gg) => gg.info.dist.q95 >= 5)
                .map((gg, idx) => (
                  <li key={idx}>
                    <CombinedGroup
                      plannedTrip={tripId}
                      combinedGroup={gg}
                      startStation={getAlternativeFromStation(sec, gg)}
                      earliestDeparture={getAlternativeEarliestDeparture(
                        sec,
                        gg,
                      )}
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
            {groupFilter !== "All" || groupByStation === "EntryAndLast" ? (
              <label className="inline-flex items-center gap-1">
                <input
                  type="checkbox"
                  name="group-by-other-trip"
                  checked={groupByOtherTrip}
                  onChange={() => setGroupByOtherTrip((val) => !val)}
                />
                {groupFilter === "Entering" || groupByStation === "EntryAndLast"
                  ? "Zubringer"
                  : "Abbringer"}
              </label>
            ) : null}
          </div>
        </div>
      </form>
      {content}
    </div>
  );
}

function getAlternativeFromStation(
  sec: GroupsInTripSection,
  gg: GroupedPassengerGroups,
): Station {
  if (gg.entry_station.length === 1) {
    return gg.entry_station[0];
  } else {
    return sec.from;
  }
}

function getAlternativeEarliestDeparture(
  sec: GroupsInTripSection,
  gg: GroupedPassengerGroups,
): number {
  if (gg.entry_station.length === 1) {
    return gg.entry_time;
  } else {
    return sec.departure_current_time;
  }
}

export default TripSectionDetails;
