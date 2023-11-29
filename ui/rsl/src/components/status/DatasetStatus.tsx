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
      <table className="border-separate border-spacing-x-2">
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
          <tr className="text-left">
            <th className="font-medium">Datei</th>
            <th className="font-medium">Änderungsdatum</th>
            <th className="font-medium">Geladene Reiseketten</th>
            <th className="font-medium">Nicht geladene Reiseketten</th>
          </tr>
        </thead>
        <tbody>
          {journey_files.map((file) => (
            <tr key={file.name}>
              <td>{file.name}</td>
              <td>{formatDateTime(file.last_modified)}</td>
              <td>{formatNumber(file.matched_journeys)}</td>
              <td>
                {formatNumber(file.unmatched_journeys)} (
                {formatPercent(
                  file.unmatched_journeys /
                    (file.matched_journeys + file.unmatched_journeys),
                )}
                )
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
      <table className="border-separate border-spacing-x-2">
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
