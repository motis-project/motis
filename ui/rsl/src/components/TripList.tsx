import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, SelectorIcon } from "@heroicons/react/solid";
import { add, fromUnixTime, getUnixTime, max, sub } from "date-fns";
import { useAtom } from "jotai";
import { Fragment, useCallback, useState } from "react";
import { useInfiniteQuery } from "react-query";
import { Virtuoso } from "react-virtuoso";

import { TripServiceInfo } from "@/api/protocol/motis";
import {
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsSortOrder,
  PaxMonFilteredTripInfo,
} from "@/api/protocol/motis/paxmon";

import { useLookupScheduleInfoQuery } from "@/api/lookup";
import { sendPaxMonFilterTripsRequest } from "@/api/paxmon";
import { ServiceClasses } from "@/api/serviceClasses";

import { formatPercent } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatISODate, formatTime } from "@/util/dateFormat";
import { getMaxPax } from "@/util/statistics";

import MiniTripLoadGraph2 from "@/components/MiniTripLoadGraph2";

type LabeledFilterOption = {
  option: PaxMonFilterTripsSortOrder;
  label: string;
};

const sortOptions: Array<LabeledFilterOption> = [
  { option: "MaxLoad", label: "Züge sortiert nach Auslastung (prozentual)" },
  {
    option: "MostCritical",
    label: "Züge sortiert nach Anzahl Reisender über Kapazität",
  },
  {
    option: "EarliestCritical",
    label: "Züge sortiert nach erstem kritischen Abschnitt",
  },
  {
    option: "FirstDeparture",
    label: "Züge sortiert nach Abfahrtszeit am ersten Halt",
  },
  { option: "TrainNr", label: "Züge sortiert nach Zugnummer" },
  { option: "ExpectedPax", label: "Züge sortiert nach Buchungen" },
];

function getFilterTripsRequest(
  universe: number,
  sortOrder: PaxMonFilterTripsSortOrder,
  selectedDate: Date | undefined,
  filterTrainNrs: number[],
  pageParam: number
): PaxMonFilterTripsRequest {
  return {
    universe,
    ignore_past_sections: false,
    include_load_threshold: 0.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: true,
    sort_by: sortOrder,
    max_results: 100,
    skip_first: pageParam,
    filter_by_time: selectedDate ? "DepartureOrArrivalTime" : "NoFilter",
    filter_interval: {
      begin: selectedDate ? getUnixTime(selectedDate) : 0,
      end: selectedDate ? getUnixTime(add(selectedDate, { days: 1 })) : 0,
    },
    filter_by_train_nr: filterTrainNrs.length > 0,
    filter_train_nrs: filterTrainNrs,
    filter_by_service_class: false,
    filter_service_classes: [],
  };
}

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

  const [selectedSort, setSelectedSort] = useState(sortOptions[0]);
  const [selectedDate, setSelectedDate] = useState<Date>();
  const [trainNrFilter, setTrainNrFilter] = useState("");

  const filterTrainNrs = [...trainNrFilter.matchAll(/\d+/g)].map((m) =>
    parseInt(m[0])
  );

  const { data: scheduleInfo } = useLookupScheduleInfoQuery();

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
      {
        universe,
        sortOrder: selectedSort.option,
        selectedDate,
        filterTrainNrs,
      },
    ],
    ({ pageParam = 0 }) => {
      const req = getFilterTripsRequest(
        universe,
        selectedSort.option,
        selectedDate,
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
      enabled: selectedDate != undefined,
    }
  );

  if (selectedDate == undefined && scheduleInfo) {
    setSelectedDate(fromUnixTime(scheduleInfo.begin));
  }
  const minDate = scheduleInfo ? fromUnixTime(scheduleInfo.begin) : undefined;
  const maxDate =
    scheduleInfo && minDate
      ? max([minDate, sub(fromUnixTime(scheduleInfo.end), { days: 1 })])
      : undefined;

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
      <Listbox value={selectedSort} onChange={setSelectedSort}>
        <div className="relative mb-2">
          <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
            <span className="block truncate">{selectedSort.label}</span>
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
              {sortOptions.map((opt) => (
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
      <div className="flex justify-between pb-2">
        <div className="w-1/2 pr-1">
          <label>
            <span className="text-sm">Datum</span>
            <input
              type="date"
              min={minDate ? formatISODate(minDate) : undefined}
              max={maxDate ? formatISODate(maxDate) : undefined}
              value={selectedDate ? formatISODate(selectedDate) : ""}
              onChange={(e) =>
                setSelectedDate(e.target.valueAsDate ?? undefined)
              }
              className="block w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
          </label>
        </div>
        <div className="w-1/2 pl-1">
          <label>
            <span className="text-sm">Zugnummer(n)</span>
            <input
              type="text"
              className="block w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              value={trainNrFilter}
              onChange={(e) => setTrainNrFilter(e.target.value)}
            />
          </label>
        </div>
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
        <div className="space-y-2">
          {/*<MiniTripLoadGraph edges={ti.edges} />*/}
          <MiniTripLoadGraph2 edges={ti.edges} />
        </div>
        {criticalInfo}
      </div>
    </div>
  );
}

export default TripList;
