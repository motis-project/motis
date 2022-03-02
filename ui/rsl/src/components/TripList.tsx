import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, SelectorIcon } from "@heroicons/react/solid";
import { useAtom } from "jotai";
import { Fragment, useState } from "react";
import { Virtuoso } from "react-virtuoso";

import { TripId } from "@/api/protocol/motis";
import {
  PaxMonFilterTripsRequest,
  PaxMonFilteredTripInfo,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonFilterTripsRequest } from "@/api/paxmon";

import { formatPercent } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatTime } from "@/util/dateFormat";
import { getMaxPax } from "@/util/statistics";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";

type FilterOption = "MostCritical" | "ByTrainNr" | "ByDeparture";

type LabeledFilterOption = { option: FilterOption; label: string };

const filterOptions: Array<LabeledFilterOption> = [
  { option: "MostCritical", label: "Kritische Züge" },
  { option: "ByTrainNr", label: "Alle Züge (nach Zugnummer)" },
  { option: "ByDeparture", label: "Alle Züge (nach Abfahrtszeit)" },
];

function getFilterTripsRequest(
  universe: number,
  filterOption: FilterOption
): PaxMonFilterTripsRequest {
  const req: PaxMonFilterTripsRequest = {
    universe,
    ignore_past_sections: true,
    include_load_threshold: 0.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: true,
    sort_by: "MostCritical",
    max_results: 100,
    skip_first: 0,
    filter_by_time: "NoFilter",
    filter_interval: { begin: 0, end: 0 },
  };

  switch (filterOption) {
    case "MostCritical":
      req.include_load_threshold = 1.0;
      break;
    case "ByTrainNr":
      req.sort_by = "TrainNr";
      break;
    case "ByDeparture":
      req.sort_by = "FirstDeparture";
      break;
  }

  return req;
}

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

  const [selectedFilter, setSelectedFilter] = useState(filterOptions[0]);

  const { data /*, isLoading, error */ } = usePaxMonFilterTripsRequest(
    getFilterTripsRequest(universe, selectedFilter.option)
  );

  const selectedTripId = JSON.stringify(selectedTrip);

  return (
    <div className="h-full flex flex-col">
      <div className="pb-3">
        <Listbox value={selectedFilter} onChange={setSelectedFilter}>
          <div className="relative mt-1">
            <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
              <span className="block truncate">{selectedFilter.label}</span>
              <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <SelectorIcon
                  className="w-5 h-5 text-gray-400"
                  aria-hidden="true"
                />
              </span>
            </Listbox.Button>
            <Transition
              as={Fragment}
              leave="transition ease-in duration-100"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <Listbox.Options className="absolute z-20 w-full py-1 mt-1 overflow-auto text-base bg-white rounded-md shadow-lg max-h-60 ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
                {filterOptions.map((opt) => (
                  <Listbox.Option
                    key={opt.option}
                    value={opt}
                    className={({ active }) =>
                      classNames(
                        "cursor-default select-none relative py-2 pl-10 pr-4",
                        active ? "text-amber-900 bg-amber-100" : "text-gray-900"
                      )
                    }
                  >
                    {({ selected, active }) => (
                      <>
                        <span
                          className={classNames(
                            "block truncate",
                            selected ? "font-medium" : "font-normal"
                          )}
                        >
                          {opt.label}
                        </span>
                        {selected ? (
                          <span
                            className={classNames(
                              "absolute inset-y-0 left-0 flex items-center pl-3",
                              active ? "text-amber-600" : "text-amber-600"
                            )}
                          >
                            <CheckIcon className="w-5 h-5" aria-hidden="true" />
                          </span>
                        ) : null}
                      </>
                    )}
                  </Listbox.Option>
                ))}
              </Listbox.Options>
            </Transition>
          </div>
        </Listbox>
      </div>
      <div className="grow">
        {data ? (
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
        ) : (
          <div>Züge werden geladen...</div>
        )}
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

  const critSections = ti.edges
    .filter((e) => e.possibly_over_capacity && e.capacity_type === "Known")
    .map((e) => {
      const maxPax = getMaxPax(e.passenger_cdf);
      return {
        edge: e,
        maxPax,
        maxPercent: maxPax / e.capacity,
        maxOverCap: Math.max(0, maxPax - e.capacity),
      };
    });
  let criticalInfo = null;

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
        {mostCritSection != firstCritSection && (
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
        )}
      </div>
    );
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
