import { ArrowDownTrayIcon } from "@heroicons/react/20/solid";
import { useQuery } from "@tanstack/react-query";
import { add, getUnixTime } from "date-fns";
import { useAtom } from "jotai";
import React, { ReactElement, useState } from "react";

import {
  PaxMonDetailedCapacityStatusRequest,
  PaxMonDetailedCapacityStatusResponse,
  PaxMonDetailedTripCapacityStats,
} from "@/api/protocol/motis/paxmon";

import { ServiceClass } from "@/api/constants";
import { getApiEndpoint } from "@/api/endpoint";
import {
  queryKeys,
  sendPaxMonDetailedCapacityStatusRequest,
} from "@/api/paxmon";
import { useRISStatusRequest } from "@/api/ris";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import { formatFileNameTime } from "@/util/dateFormat";

import ServiceClassFilter from "@/components/inputs/ServiceClassFilter";
import Baureihe from "@/components/util/Baureihe";

export interface DetailedCapacityStatusProps {
  selectedDate: Date | undefined | null;
}

function DetailedCapacityStatus({
  selectedDate,
}: DetailedCapacityStatusProps): ReactElement {
  const [universe] = useAtom(universeAtom);
  const [serviceClassFilter, setServiceClassFilter] = useState([
    ServiceClass.ICE,
    ServiceClass.IC,
  ]);

  const request: PaxMonDetailedCapacityStatusRequest = {
    universe,
    filter_by_time: selectedDate ? "ActiveTime" : "NoFilter",
    filter_interval: {
      begin: selectedDate ? getUnixTime(selectedDate) : 0,
      end: selectedDate ? getUnixTime(add(selectedDate, { days: 1 })) : 0,
    },
    include_missing_vehicle_infos: true,
    include_uics_not_found: false,
  };

  const { data } = useQuery(
    queryKeys.detailedCapacityStatus(request),
    () => sendPaxMonDetailedCapacityStatusRequest(request),
    {
      enabled: selectedDate !== undefined,
    },
  );

  return (
    <div className="py-3">
      <div className="flex pb-2 gap-1">
        <div className="flex flex-col justify-end">
          <ServiceClassFilter
            selectedServiceClasses={serviceClassFilter}
            setSelectedServiceClasses={setServiceClassFilter}
            popupPosition="left-0"
          />
        </div>
      </div>
      <CapacityStatusDisplay
        data={data}
        serviceClassFilter={serviceClassFilter}
      />
      <CsvDownloadButtons />
    </div>
  );
}

interface CapacityStatusDisplayProps {
  data: PaxMonDetailedCapacityStatusResponse | undefined;
  serviceClassFilter: ServiceClass[];
}

interface CapacityStatusDataProps {
  data: PaxMonDetailedCapacityStatusResponse;
  serviceClassFilter: ServiceClass[];
}

function CapacityStatusDisplay({
  data,
  serviceClassFilter,
}: CapacityStatusDisplayProps) {
  if (!data) {
    return <div>Daten werden geladen...</div>;
  }

  return (
    <>
      <CapacityStatusStats
        data={data}
        serviceClassFilter={serviceClassFilter}
      />
      <MissingVehicles data={data} />
    </>
  );
}

function CapacityStatusStats({
  data,
  serviceClassFilter,
}: CapacityStatusDataProps) {
  interface Column {
    label: string;
    stats: PaxMonDetailedTripCapacityStats;
  }

  const columns: Column[] = [
    { label: "Alle Züge", stats: data.all_trips },
    ...data.by_category
      .filter((c) => serviceClassFilter.includes(c.service_class))
      .map((c) => {
        return { label: c.category, stats: c };
      }),
  ];

  return (
    <div>
      <table className="border-separate border-spacing-x-2">
        <thead>
          <tr className="text-left">
            <th className="font-medium"></th>
            {columns.map((c) => (
              <th key={c.label} className="font-medium text-center p-1">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="font-medium">Überwachte Züge</td>
            {columns.map((c) => (
              <td key={c.label} className="text-center p-1">
                {formatNumber(c.stats.tracked)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Kapazitätsdaten für alle Abschnitte</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.capacity_for_all_sections}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Kapazitätsdaten vollständig vorhanden
            </td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.full_data}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">Kapazitätsdaten teilweise vorhanden</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.partial_data}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">Wagenreihungsdaten vorhanden</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.trip_formation_data_found}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">Keinerlei Wagenreihungsdaten</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.no_formation_data_at_all}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Wagenreihungen auf einigen Abschnitten (alle{" "}
              <abbr title="vereinigte Züge">V.Z.</abbr>)
            </td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.no_formation_data_some_sections_all_merged}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Wagenreihungen auf einigen Abschnitten (einige{" "}
              <abbr title="vereinigte Züge">V.Z.</abbr>)
            </td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.no_formation_data_some_sections_some_merged}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">Keinerlei Wagen gefunden</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.no_vehicles_found_at_all}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Wagen gefunden auf einigen Abschnitten
            </td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.no_vehicles_found_some_sections}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Einige Wagen nicht gefunden auf einigen Abschnitten
            </td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.some_vehicles_not_found_some_sections}
                total={c.stats.tracked}
              />
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

interface StatsTableCellProps {
  value: number;
  total: number;
}

function StatsTableCell({ value, total }: StatsTableCellProps) {
  return (
    <td
      className="text-center p-1"
      title={`${formatNumber(value)} von ${formatNumber(total)}`}
    >
      {formatPercent(value / total, {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      })}
    </td>
  );
}

interface MissingVehiclesProps {
  data: PaxMonDetailedCapacityStatusResponse;
}

function MissingVehicles({ data }: MissingVehiclesProps) {
  return (
    <div className="pt-3">
      <details>
        <summary className="cursor-pointer select-none">
          Nicht gefundene Wagen nach Bauart und Baureihe
        </summary>
        <table className="border-separate border-spacing-x-2">
          <thead>
            <tr className="text-left">
              <th className="font-medium">Anzahl</th>
              <th className="font-medium">
                <a
                  href="https://de.wikipedia.org/wiki/UIC-Bauart-Bezeichnungssystem_f%C3%BCr_Reisezugwagen"
                  target="_blank"
                  rel="noreferrer"
                  referrerPolicy="no-referrer"
                  className="underline decoration-dotted"
                >
                  Bauart
                </a>
              </th>
              <th className="font-medium">Baureihe</th>
            </tr>
          </thead>
          <tbody>
            {data.missing_vehicle_infos.map((vi) => (
              <tr key={`${vi.type_code} ${vi.baureihe}`}>
                <td>{formatNumber(vi.count)}</td>
                <td>{vi.type_code}</td>
                <td>
                  <Baureihe baureihe={vi.baureihe} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

function CsvDownloadButtons() {
  const { data } = useRISStatusRequest();

  const suffix = data ? "-" + formatFileNameTime(data.system_time) : "";

  return (
    <div className="flex gap-3 pt-5">
      <a
        href={`${getApiEndpoint()}paxmon/capacity_status/trips.csv`}
        download={`rsl-trips${suffix}.csv`}
        className="inline-flex items-center gap-3 px-3 py-1 rounded text-white bg-db-red-500 hover:bg-db-red-600"
      >
        <ArrowDownTrayIcon className="h-5 w-5" />
        Liste überwachter Züge (CSV)
      </a>
      <a
        href={`${getApiEndpoint()}paxmon/capacity_status/formations.csv`}
        download={`rls-formations${suffix}.csv`}
        className="inline-flex items-center gap-3 px-3 py-1 rounded text-white bg-db-red-500 hover:bg-db-red-600"
      >
        <ArrowDownTrayIcon className="h-5 w-5" />
        Wagenreihungen (CSV)
      </a>
    </div>
  );
}

export default DetailedCapacityStatus;
