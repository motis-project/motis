import { useAtom } from "jotai";

import { TripId } from "@/api/protocol/motis";

import { usePaxMonFilterTripsRequest } from "@/api/paxmon";

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
    max_results: 100,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
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
                Reisende über Kapazität: {ti.max_excess_pax} maximal,{" "}
                {ti.cumulative_excess_pax} gesamt
              </div>
            </div>
          </div>
        ))}
      </div>
      {data.total_matching_trips > data.filtered_trips ? (
        <div>
          ...und {data.total_matching_trips - data.filtered_trips} weitere
          kritische Züge ({data.total_matching_trips} insgesamt)
        </div>
      ) : null}
    </div>
  );
}

export default CriticalTripList;
