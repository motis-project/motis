import { useAtom } from "jotai/index";
import { ReactElement } from "react";

import {
  PaxMonCapacityStatusResponse,
  PaxMonTripCapacityStats,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonCapacityStatus } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

function CapacityStatus(): ReactElement {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonCapacityStatus({
    universe,
    include_trips_without_capacity: false,
    include_other_trips_without_capacity: false,
    include_missing_vehicle_infos: false,
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

function CapacityStatusDisplay({ data }: CapacityStatusDisplayProps) {
  if (!data) {
    return <div>Daten werden geladen...</div>;
  }

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

export default CapacityStatus;
