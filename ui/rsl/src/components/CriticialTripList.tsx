import { useAtom } from "jotai";

import { TripId } from "@/api/protocol/motis";

import { usePaxMonFilterTripsRequest } from "@/api/paxmon";

import { formatNumber } from "@/data/numberFormat";
import { universeAtom } from "@/data/simulation";

import TripServiceInfoView from "@/components/TripServiceInfoView";

export type CriticalTripListProps = {
  onTripSelected: (trip: TripId | undefined) => void;
};

function CriticalTripList({
  onTripSelected,
}: CriticalTripListProps): JSX.Element {
  const [universe] = useAtom(universeAtom);

  const { data /*, isLoading, error */ } = usePaxMonFilterTripsRequest({
    universe,
    ignore_past_sections: true,
    include_load_threshold: 1.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: false,
    sort_by: "MostCritical",
    max_results: 100,
    skip_first: 0,
    filter_by_time: "NoFilter",
    filter_interval: { begin: 0, end: 0 },
  });

  if (!data) {
    return <div>Kritische Züge werden geladen...</div>;
  }

  return (
    <div>
      <div className="mb-4 text-lg font-semibold">Kritische Züge:</div>
      <div className="flex flex-col gap-4">
        {data.trips.map((ti) => (
          <div
            key={JSON.stringify(ti.tsi.trip)}
            className="cursor-pointer"
            onClick={() => onTripSelected(ti.tsi.trip)}
          >
            <TripServiceInfoView tsi={ti.tsi} format="Long" />
            <div>
              <div>
                Kritische Abschnitte: {ti.critical_sections}/{ti.section_count}{" "}
              </div>
              <div>
                Reisende über Kapazität: {formatNumber(ti.max_excess_pax)} max.,{" "}
                {formatNumber(ti.cumulative_excess_pax)} gesamt
              </div>
            </div>
          </div>
        ))}
      </div>
      {data.remaining_trips > 0 ? (
        <div>
          ...und {formatNumber(data.remaining_trips)} weitere kritische Züge (
          {formatNumber(data.total_matching_trips)} insgesamt)
        </div>
      ) : null}
    </div>
  );
}

export default CriticalTripList;
