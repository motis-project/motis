import { useAtom } from "jotai";
import { Virtuoso } from "react-virtuoso";

import { TripId } from "@/api/protocol/motis";
import { PaxMonFilteredTripInfo } from "@/api/protocol/motis/paxmon";

import { usePaxMonFilterTripsRequest } from "@/api/paxmon";

import { formatNumber, formatPercent } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatTime } from "@/util/dateFormat";
import { getMaxPax } from "@/util/statistics";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

  const { data /*, isLoading, error */ } = usePaxMonFilterTripsRequest({
    universe,
    ignore_past_sections: true,
    include_load_threshold: 1.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: true,
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

  const category = ti.tsi.service_infos[0]?.category ?? "";
  const trainNr = ti.tsi.service_infos[0]?.train_nr ?? ti.tsi.trip.train_nr;

  let criticalInfo = null;
  if (ti.critical_sections !== 0) {
    const critSections = ti.edges
      .filter((e) => e.possibly_over_capacity)
      .map((e) => {
        const maxPax = getMaxPax(e.passenger_cdf);
        return {
          edge: e,
          maxPax,
          maxPercent: maxPax / e.capacity,
          maxOverCap: Math.max(0, maxPax - e.capacity),
        };
      });
    if (critSections.length > 0) {
      const firstCritSection = critSections[0];
      const mostCritSection = critSections.sort(
        (a, b) => b.maxPercent - a.maxPercent
      )[0];

      criticalInfo = (
        <div className="pt-1 flex flex-col gap-1">
          <div>
            <div className="text-xs">Kritisch ab:</div>
            <div className="flex justify-between">
              <div className="truncate">
                {firstCritSection.edge.from.name} (
                {formatTime(firstCritSection.edge.departure_schedule_time)})
              </div>
              <div>{formatPercent(firstCritSection.maxPercent)}</div>
            </div>
          </div>
          <div>
            <div className="text-xs">Kritischster Abschnitt ab:</div>
            <div className="flex justify-between">
              <div className="truncate">
                {mostCritSection.edge.from.name} (
                {formatTime(mostCritSection.edge.departure_schedule_time)})
              </div>
              <div>{formatPercent(mostCritSection.maxPercent)}</div>
            </div>
          </div>
        </div>
      );
    }
  }

  return (
    <div className="pr-1 pb-3">
      <div
        className={classNames(
          "cursor-pointer p-1 rounded",
          isSelected ? "bg-db-cool-gray-300 shadow-md" : "bg-db-cool-gray-100"
        )}
        onClick={() => setSelectedTrip(ti.tsi.trip)}
      >
        <div className="flex gap-4 pb-1">
          <div className="flex flex-col">
            <div className="text-sm text-center">{category}</div>
            <div className="text-xl font-semibold">{trainNr}</div>
          </div>
          <div className="grow flex flex-col truncate">
            <div className="flex justify-between">
              <div className="truncate">{ti.tsi.primary_station.name}</div>
              <div>{formatTime(ti.tsi.trip.time)}</div>
            </div>
            <div className="flex justify-between">
              <div className="truncate">{ti.tsi.secondary_station.name}</div>
              <div>{formatTime(ti.tsi.trip.target_time)}</div>
            </div>
          </div>
        </div>
        <MiniTripLoadGraph edges={ti.edges} />
        {criticalInfo}
      </div>
    </div>
  );
}

export default TripList;
