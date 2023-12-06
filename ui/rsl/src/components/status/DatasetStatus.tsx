import { ReactElement } from "react";

import {
  PaxMonCapacityFileFormat,
  PaxMonCapacityFileInfo,
  PaxMonJourneyFileInfo,
  PaxMonScheduleInfo,
} from "@/api/protocol/motis/paxmon.ts";

import { usePaxMonDatasetInfo } from "@/api/paxmon.ts";

import { formatNumber, formatPercent } from "@/data/numberFormat.ts";

import { formatDateTime } from "@/util/dateFormat.ts";

function DatasetStatus(): ReactElement {
  const { data: datasetInfo } = usePaxMonDatasetInfo();

  if (!datasetInfo) {
    return <div>Laden...</div>;
  }

  return (
    <>
      <ScheduleInfo schedule={datasetInfo.schedule} />
      <JourneyFilesInfo journey_files={datasetInfo.journey_files} />
      <CapacityFilesInfo capacity_files={datasetInfo.capacity_files} />
    </>
  );
}

function ScheduleInfo({ schedule }: { schedule: PaxMonScheduleInfo }) {
  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Sollfahrplan</h2>
      <table className="-ml-2 border-separate border-spacing-x-4">
        <tbody>
          <tr>
            <td className="font-medium">Fahrplanbezeichnung</td>
            <td>
              {schedule.names.map((name) => (
                <div key={name}>{name}</div>
              ))}
            </td>
          </tr>
          <tr>
            <td className="font-medium">Geladener Ausschnitt</td>
            <td>
              {formatDateTime(schedule.begin)} bis{" "}
              {formatDateTime(schedule.end)}
            </td>
          </tr>
          <tr>
            <td className="font-medium">Geladene Stationen</td>
            <td>{formatNumber(schedule.station_count)}</td>
          </tr>
          <tr>
            <td className="font-medium">Geladene Züge</td>
            <td>
              {formatNumber(schedule.trip_count)}{" "}
              <span>(inkl. Echtzeitupdates)</span>
            </td>
          </tr>
          <tr>
            <td className="font-medium">Vereinigungen &amp; Durchbindungen</td>
            <td>
              {schedule.expanded_trip_count > 0 ? "Aktiviert" : "Deaktiviert"}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function JourneyFilesInfo({
  journey_files,
}: {
  journey_files: PaxMonJourneyFileInfo[];
}) {
  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Reiseketten</h2>
      <table className="border-separate border-spacing-x-2">
        <thead>
          <tr className="text-center">
            <th colSpan={2}></th>
            <th colSpan={3} className="pl-5 font-medium">
              Reiseketten
            </th>
            <th colSpan={3} className="pl-5 font-medium">
              Reisendengruppen
            </th>
            <th colSpan={3} className="pl-5 font-medium">
              Reisende
            </th>
          </tr>
          <tr className="text-left">
            <th className="font-medium">Datei</th>
            <th className="pl-5 font-medium">Änderungsdatum</th>
            <th className="pl-5 font-medium">Geladen</th>
            <th className="font-medium">Neu berechnet</th>
            <th className="font-medium">Ignoriert</th>
            <th className="pl-5 font-medium">Geladen</th>
            <th className="font-medium">Neu berechnet</th>
            <th className="font-medium">Ignoriert</th>
            <th className="pl-5 font-medium">Geladen</th>
            <th className="font-medium">Neu berechnet</th>
            <th className="font-medium">Ignoriert</th>
          </tr>
        </thead>
        <tbody>
          {journey_files.map((file) => (
            <JourneyFileRow file={file} key={file.name} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JourneyFileRow({ file }: { file: PaxMonJourneyFileInfo }) {
  const totalJourneys = file.matched_journeys + file.unmatched_journeys;
  const loadedJourneys =
    file.matched_journeys + file.unmatched_journeys_rerouted;
  const skippedJourneys =
    file.unmatched_journeys - file.unmatched_journeys_rerouted;

  const totalGroups = file.matched_groups + file.unmatched_groups;
  const loadedGroups = file.matched_groups + file.unmatched_groups_rerouted;
  const skippedGroups = file.unmatched_groups - file.unmatched_groups_rerouted;

  const totalPax = file.matched_pax + file.unmatched_pax;
  const loadedPax = file.matched_pax + file.unmatched_pax_rerouted;
  const skippedPax = file.unmatched_pax - file.unmatched_pax_rerouted;

  return (
    <tr>
      <td>{file.name}</td>
      <td className="pl-5">{formatDateTime(file.last_modified)}</td>
      <td className="pl-5">{formatNumber(loadedJourneys)}</td>
      <td>
        {formatNumber(file.unmatched_journeys_rerouted)} (
        {formatPercent(file.unmatched_journeys_rerouted / totalJourneys)})
      </td>
      <td>
        {formatNumber(skippedJourneys)} (
        {formatPercent(skippedJourneys / totalJourneys)})
      </td>
      <td className="pl-5">{formatNumber(loadedGroups)}</td>
      <td>
        {formatNumber(file.unmatched_groups_rerouted)} (
        {formatPercent(file.unmatched_groups_rerouted / totalGroups)})
      </td>
      <td>
        {formatNumber(skippedGroups)} (
        {formatPercent(skippedGroups / totalGroups)})
      </td>
      <td className="pl-5">{formatNumber(loadedPax)}</td>
      <td>
        {formatNumber(file.unmatched_pax_rerouted)} (
        {formatPercent(file.unmatched_pax_rerouted / totalPax)})
      </td>
      <td>
        {formatNumber(skippedPax)} ({formatPercent(skippedPax / totalPax)})
      </td>
    </tr>
  );
}

function getCapacityFormatName(format: PaxMonCapacityFileFormat): string {
  switch (format) {
    case "TRIP":
      return "Zugkapazitäten";
    case "RIS_SERVICE_VEHICLES":
      return "RIS Service Vehicles";
    case "FZG_KAP":
      return "Fahrzeugkapazitäten";
    case "FZG_GRUPPE":
      return "Fahrzeuggruppenkapazitäten";
    case "GATTUNG":
      return "Gattungskapazitäten";
    case "BAUREIHE":
      return "Baureihenkapazitäten";
    default:
      return format;
  }
}

function CapacityFilesInfo({
  capacity_files,
}: {
  capacity_files: PaxMonCapacityFileInfo[];
}) {
  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Kapazitätsdaten</h2>
      <table className="-ml-2 border-separate border-spacing-x-4">
        <thead>
          <tr className="text-left">
            <th className="font-medium">Datei</th>
            <th className="font-medium">Änderungsdatum</th>
            <th className="font-medium">Format</th>
            <th className="font-medium">Geladene Einträge</th>
            <th className="font-medium">Nicht geladene Einträge</th>
          </tr>
        </thead>
        <tbody>
          {capacity_files.map((file) => (
            <tr key={file.name}>
              <td>{file.name}</td>
              <td>{formatDateTime(file.last_modified)}</td>
              <td>{getCapacityFormatName(file.format)}</td>
              <td>{formatNumber(file.loaded_entry_count)}</td>
              <td>{formatNumber(file.skipped_entry_count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DatasetStatus;
