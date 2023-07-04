import { ArrowDownTrayIcon } from "@heroicons/react/20/solid";
import { useAtom } from "jotai/index";
import { ReactElement } from "react";

import {
  PaxMonCapacityStatusResponse,
  PaxMonTripCapacityStats,
} from "@/api/protocol/motis/paxmon";

import { getApiEndpoint } from "@/api/endpoint";
import { usePaxMonCapacityStatus } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import Baureihe from "@/components/util/Baureihe";

function CapacityStatus(): ReactElement {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonCapacityStatus({
    universe,
    include_missing_vehicle_infos: true,
    include_uics_not_found: false,
  });

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Kapazitätsdaten</h2>
      <CapacityStatusDisplay data={data} />
      <div className="flex gap-3 pt-5">
        <a
          href={`${getApiEndpoint()}paxmon/capacity_status/trips.csv`}
          className="inline-flex items-center gap-3 px-3 py-1 rounded text-white bg-db-red-500 hover:bg-db-red-600"
        >
          <ArrowDownTrayIcon className="h-5 w-5" />
          Liste überwachter Züge (CSV)
        </a>
        <a
          href={`${getApiEndpoint()}paxmon/capacity_status/formations.csv`}
          className="inline-flex items-center gap-3 px-3 py-1 rounded text-white bg-db-red-500 hover:bg-db-red-600"
        >
          <ArrowDownTrayIcon className="h-5 w-5" />
          Wagenreihungen (CSV)
        </a>
      </div>
    </div>
  );
}

type CapacityStatusDisplayProps = {
  data: PaxMonCapacityStatusResponse | undefined;
};

type CapacityStatusDataProps = {
  data: PaxMonCapacityStatusResponse;
};

function CapacityStatusDisplay({ data }: CapacityStatusDisplayProps) {
  if (!data) {
    return <div>Daten werden geladen...</div>;
  }

  return (
    <>
      <CapacityStatusStats data={data} />
      <MissingVehicles data={data} />
    </>
  );
}

function CapacityStatusStats({ data }: CapacityStatusDataProps) {
  type Column = { label: string; stats: PaxMonTripCapacityStats };

  const columns: Column[] = [
    { label: "Alle Züge", stats: data.all_trips },
    ...data.by_category
      .filter((c) => c.service_class <= 2)
      .map((c) => {
        return { label: c.category, stats: c };
      }),
  ];

  const numWithPercent = (c: Column, n: number) =>
    `${formatNumber(n)} (${formatPercent(n / c.stats.tracked, {
      minimumFractionDigits: 2,
    })})`;

  return (
    <div>
      <table className="border-separate border-spacing-x-2">
        <thead>
          <tr className="text-left">
            <th className="font-medium"></th>
            {columns.map((c) => (
              <th key={c.label} className="font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="font-medium">Überwachte Züge</td>
            {columns.map((c) => (
              <td key={c.label}>{formatNumber(c.stats.tracked)}</td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Kapazitätsdaten vollständig</td>
            {columns.map((c) => (
              <td key={c.label}>{numWithPercent(c, c.stats.full_data)}</td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Kapazitätsdaten teilweise</td>
            {columns.map((c) => (
              <td key={c.label}>{numWithPercent(c, c.stats.full_data)}</td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Formationsdaten vorhanden</td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(c, c.stats.trip_formation_data_found)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Keinerlei Formationsdaten</td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(c, c.stats.no_formation_data_at_all)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Formationsdaten auf einigen Abschnitten (alle{" "}
              <abbr title="vereinigte Züge">V.Z.</abbr>)
            </td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(
                  c,
                  c.stats.no_formation_data_some_sections_all_merged
                )}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Formationsdaten auf einigen Abschnitten (einige{" "}
              <abbr title="vereinigte Züge">V.Z.</abbr>)
            </td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(
                  c,
                  c.stats.no_formation_data_some_sections_some_merged
                )}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Keinerlei Wagen gefunden</td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(c, c.stats.no_vehicles_found_at_all)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Keine Wagen gefunden auf einigen Abschnitten
            </td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(c, c.stats.no_vehicles_found_some_sections)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">
              Einige Wagen nicht gefunden auf einigen Abschnitten
            </td>
            {columns.map((c) => (
              <td key={c.label}>
                {numWithPercent(
                  c,
                  c.stats.some_vehicles_not_found_some_sections
                )}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function MissingVehicles({ data }: CapacityStatusDataProps) {
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

export default CapacityStatus;
