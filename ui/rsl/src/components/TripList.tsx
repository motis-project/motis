import { useAtom } from "jotai";
import { Virtuoso } from "react-virtuoso";

import { TripId } from "@/api/protocol/motis";
import { PaxMonFilteredTripInfo } from "@/api/protocol/motis/paxmon";

import { usePaxMonFilterTripsRequest } from "@/api/paxmon";

import { formatNumber } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";

import TripServiceInfoView from "@/components/TripServiceInfoView";

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

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
    return <div>Züge werden geladen...</div>;
  }

  const selectedTripId = JSON.stringify(selectedTrip);

  return (
    <div className="h-full flex flex-col">
      <div className="mb-4 text-lg font-semibold">Kritische Züge:</div>
      <div className="grow">
        <Virtuoso
          data={data.trips}
          overscan={200}
          itemContent={(index, ti) => (
            <TripListEntry
              ti={ti}
              selectedTripId={selectedTripId}
              setSelectedTrip={setSelectedTrip}
            />
          )}
        />
      </div>
    </div>
  );
}

type TripListEntryProps = {
  ti: PaxMonFilteredTripInfo;
  selectedTripId: string | undefined;
  setSelectedTrip: (tripId: TripId) => void;
};

function TripListEntry({
  ti,
  selectedTripId,
  setSelectedTrip,
}: TripListEntryProps): JSX.Element {
  const isSelected = selectedTripId === JSON.stringify(ti.tsi.trip);

  return (
    <div className="pr-1 pb-4">
      <div
        className={classNames(
          "cursor-pointer p-1",
          isSelected && "bg-db-cool-gray-300"
        )}
        onClick={() => setSelectedTrip(ti.tsi.trip)}
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
    </div>
  );
}

export default TripList;
