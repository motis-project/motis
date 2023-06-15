import { useAtom } from "jotai/index";
import { ReactElement } from "react";

import {
  PaxMonCapacityStatusResponse,
  PaxMonTripCapacityStats,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonCapacityStatus } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import Baureihe from "@/components/util/Baureihe";

function CapacityStatus(): ReactElement {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonCapacityStatus({
    universe,
    include_trips_without_capacity: false,
    include_other_trips_without_capacity: false,
    include_missing_vehicle_infos: true,
    include_uics_not_found: false,
  });

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Kapazitätsdaten</h2>
      <CapacityStatusDisplay data={data} />
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
    { label: "Hochgeschwindigkeitszüge", stats: data.high_speed_rail_trips },
    { label: "Fernzüge", stats: data.long_distance_trips },
    { label: "Sonstige Züge", stats: data.other_trips },
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
            <td className="font-medium">Kapazitätsdaten vorhanden</td>
            {columns.map((c) => (
              <td key={c.label}>{numWithPercent(c, c.stats.ok)}</td>
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
