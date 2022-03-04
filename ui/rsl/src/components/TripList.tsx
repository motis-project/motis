import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, SelectorIcon } from "@heroicons/react/solid";
import { useAtom } from "jotai";
import { Fragment, useCallback, useState } from "react";
import { useInfiniteQuery } from "react-query";
import { Virtuoso } from "react-virtuoso";

import { TripServiceInfo } from "@/api/protocol/motis";
import {
  PaxMonFilterTripsRequest,
  PaxMonFilteredTripInfo,
} from "@/api/protocol/motis/paxmon";

import { sendPaxMonFilterTripsRequest } from "@/api/paxmon";
import { ServiceClasses } from "@/api/serviceClasses";

import { formatPercent } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatTime } from "@/util/dateFormat";
import { getMaxPax } from "@/util/statistics";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";

type FilterOption =
  | "MostCritical"
  | "ByMaxLoad"
  | "ByEarliestCritical"
  | "ByTrainNr"
  | "ByDeparture"
  | "ByExpectedPax";

type LabeledFilterOption = { option: FilterOption; label: string };

const filterOptions: Array<LabeledFilterOption> = [
  { option: "ByMaxLoad", label: "Züge sortiert nach Auslastung (prozentual)" },
  {
    option: "MostCritical",
    label: "Züge sortiert nach Anzahl Reisender über Kapazität",
  },
  {
    option: "ByEarliestCritical",
    label: "Züge sortiert nach erstem kritischen Abschnitt",
  },
  {
    option: "ByDeparture",
    label: "Züge sortiert nach Abfahrtszeit am ersten Halt",
  },
  { option: "ByTrainNr", label: "Züge sortiert nach Zugnummer" },
  { option: "ByExpectedPax", label: "Züge sortiert nach Buchungen" },
];

function getFilterTripsRequest(
  universe: number,
  filterOption: FilterOption,
  filterTrainNrs: number[],
  pageParam: number
): PaxMonFilterTripsRequest {
  const req: PaxMonFilterTripsRequest = {
    universe,
    ignore_past_sections: false,
    include_load_threshold: 0.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: true,
    sort_by: "MostCritical",
    max_results: 100,
    skip_first: pageParam,
    filter_by_time: "NoFilter",
    filter_interval: { begin: 0, end: 0 },
    filter_by_train_nr: filterTrainNrs.length > 0,
    filter_train_nrs: filterTrainNrs,
    filter_by_service_class: true,
    filter_service_classes: [ServiceClasses.ICE, ServiceClasses.IC],
  };

  switch (filterOption) {
    case "MostCritical":
      req.sort_by = "MostCritical";
      break;
    case "ByMaxLoad":
      req.sort_by = "MaxLoad";
      break;
    case "ByEarliestCritical":
      req.sort_by = "EarliestCritical";
      break;
    case "ByTrainNr":
      req.sort_by = "TrainNr";
      break;
    case "ByDeparture":
      req.sort_by = "FirstDeparture";
      break;
    case "ByExpectedPax":
      req.sort_by = "ExpectedPax";
      break;
  }

  return req;
}

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

  const [selectedFilter, setSelectedFilter] = useState(filterOptions[0]);
  const [trainNrFilter, setTrainNrFilter] = useState("");

  const filterTrainNrs = [...trainNrFilter.matchAll(/\d+/g)].map((m) =>
    parseInt(m[0])
  );

  const {
    data,
    fetchNextPage,
    hasNextPage,
    /*
    error,
    isFetching,
    isFetchingNextPage,
    status,
    isLoading,
    isStale,
    isPreviousData,
    */
  } = useInfiniteQuery(
    [
      "tripList",
      { universe, filterOption: selectedFilter.option, filterTrainNrs },
    ],
    ({ pageParam = 0 }) => {
      const req = getFilterTripsRequest(
        universe,
        selectedFilter.option,
        filterTrainNrs,
        pageParam
      );
      return sendPaxMonFilterTripsRequest(req);
    },
    {
      getNextPageParam: (lastPage) =>
        lastPage.remaining_trips > 0 ? lastPage.next_skip : undefined,
      refetchOnWindowFocus: false,
      keepPreviousData: true,
      staleTime: 60000,
    }
  );

  const loadMore = useCallback(() => {
    if (hasNextPage) {
      return fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage]);

  const allTrips: PaxMonFilteredTripInfo[] = data
    ? data.pages.flatMap((p) => p.trips)
    : [];
  const totalNumberOfTrips = data?.pages[0]?.total_matching_trips;

  const selectedTripId = JSON.stringify(selectedTrip?.trip);

  return (
    <div className="h-full flex flex-col">
      <Listbox value={selectedFilter} onChange={setSelectedFilter}>
        <div className="relative mb-2">
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
      <div className="flex items-center gap-2 mb-4">
        <div className="text-sm">Zugnummer(n):</div>
        <input
          type="text"
          className="block w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
          value={trainNrFilter}
          onChange={(e) => setTrainNrFilter(e.target.value)}
        />
      </div>
      {totalNumberOfTrips !== undefined && (
        <div className="pb-2 text-lg">
          {totalNumberOfTrips} {totalNumberOfTrips === 1 ? "Zug" : "Züge"}
        </div>
      )}
      <div className="grow">
        {data ? (
          <Virtuoso
            data={allTrips}
            increaseViewportBy={500}
            endReached={loadMore}
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
  setSelectedTrip: (tsi: TripServiceInfo) => void;
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
          <div className="flex justify-between">
            <div className="text-xs">Kritisch ab:</div>
            <div className="text-xs">
              {firstCritSection.maxOverCap} über Kapazität
            </div>
          </div>
          <div className="flex justify-between">
            <div className="space-x-1 truncate">
              <span>
                {formatTime(firstCritSection.edge.departure_schedule_time)}
              </span>
              <span>{firstCritSection.edge.from.name}</span>
            </div>
            <div className="whitespace-nowrap">
              {formatPercent(firstCritSection.maxPercent)}
            </div>
          </div>
        </div>
        {mostCritSection != firstCritSection && (
          <div>
            <div className="flex justify-between">
              <div className="text-xs">Kritischster Abschnitt ab:</div>
              <div className="text-xs">
                {mostCritSection.maxOverCap} über Kapazität
              </div>
            </div>
            <div className="flex justify-between">
              <div className="space-x-1 truncate">
                <span>
                  {formatTime(mostCritSection.edge.departure_schedule_time)}
                </span>
                <span>{mostCritSection.edge.from.name}</span>
              </div>
              <div className="whitespace-nowrap">
                {formatPercent(mostCritSection.maxPercent)}
              </div>
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
        onClick={() => setSelectedTrip(ti.tsi)}
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
