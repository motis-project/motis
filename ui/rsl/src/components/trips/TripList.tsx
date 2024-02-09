import { Listbox, Transition } from "@headlessui/react";
import {
  ArrowPathIcon,
  CheckIcon,
  ChevronUpDownIcon,
} from "@heroicons/react/20/solid";
import { keepPreviousData, useInfiniteQuery } from "@tanstack/react-query";
import { useAtom } from "jotai";
import React, { Fragment, useCallback, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Virtuoso } from "react-virtuoso";

import {
  PaxMonEdgeLoadInfo,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsSortOrder,
  PaxMonFilteredTripInfo,
} from "@/api/protocol/motis/paxmon";

import { ServiceClass } from "@/api/constants";
import { useLookupScheduleInfoQuery } from "@/api/lookup";
import { sendPaxMonFilterTripsRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import { formatTime } from "@/util/dateFormat";
import { extractNumbers } from "@/util/extractNumbers";
import { getDayInterval } from "@/util/interval";
import { getScheduleRange } from "@/util/scheduleRange";

import DatePicker from "@/components/inputs/DatePicker";
import ServiceClassFilter from "@/components/inputs/ServiceClassFilter";
import MiniTripLoadGraph from "@/components/trips/MiniTripLoadGraph";

import { cn } from "@/lib/utils";

interface LabeledFilterOption {
  option: PaxMonFilterTripsSortOrder;
  label: string;
}

const sortOptions: LabeledFilterOption[] = [
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
  { option: "MaxPaxRange", label: "Züge sortiert nach Unsicherheit" },
  { option: "MaxPax", label: "Züge sortiert nach Anzahl Reisender" },
  { option: "MaxCapacity", label: "Züge sortiert nach Kapazität" },
  { option: "ExpectedPax", label: "Züge sortiert nach erwarteten Reisenden" },
  {
    option: "BiggestDeviation",
    label: "Züge sortiert nach größter Abweichung",
  },
  {
    option: "SmallestDeviation",
    label: "Züge sortiert nach geringster Abweichung",
  },
];

function getFilterTripsRequest(
  universe: number,
  sortOrder: PaxMonFilterTripsSortOrder,
  selectedDate: Date | undefined | null,
  filterTrainNrs: number[],
  pageParam: number,
  serviceClassFilter: number[],
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
    filter_by_time: selectedDate ? "ActiveTime" : "NoFilter",
    filter_interval: getDayInterval(selectedDate),
    filter_by_train_nr: filterTrainNrs.length > 0,
    filter_train_nrs: filterTrainNrs,
    filter_by_service_class: true,
    filter_service_classes: serviceClassFilter,
    filter_by_capacity_status: false,
    filter_has_trip_formation: false,
    filter_has_capacity_for_all_sections: false,
  };
}

function TripList(): JSX.Element {
  const params = useParams();
  const [universe] = useAtom(universeAtom);

  const [selectedSort, setSelectedSort] = useState(sortOptions[0]);
  const [selectedDate, setSelectedDate] = useState<Date | undefined | null>();
  const [trainNrFilter, setTrainNrFilter] = useState("");
  const [serviceClassFilter, setServiceClassFilter] = useState([
    ServiceClass.ICE,
    ServiceClass.IC,
  ]);

  const filterTrainNrs = extractNumbers(trainNrFilter);

  const { data: scheduleInfo } = useLookupScheduleInfoQuery();

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetching,
    refetch,
    /*
    error,
    isFetching,
    isFetchingNextPage,
    status,
    isLoading,
    isStale,
    isPreviousData,
    */
  } = useInfiniteQuery({
    queryKey: [
      "tripList",
      {
        universe,
        sortOrder: selectedSort.option,
        selectedDate,
        filterTrainNrs,
        serviceClassFilter,
      },
    ],
    queryFn: ({ pageParam }) => {
      const req = getFilterTripsRequest(
        universe,
        selectedSort.option,
        selectedDate,
        filterTrainNrs,
        pageParam,
        serviceClassFilter,
      );
      return sendPaxMonFilterTripsRequest(req);
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage) =>
      lastPage.remaining_trips > 0 ? lastPage.next_skip : undefined,
    refetchOnWindowFocus: false,
    placeholderData: keepPreviousData,
    staleTime: 60000,
    enabled: selectedDate !== undefined,
  });

  const loadMore = useCallback(() => {
    if (hasNextPage) {
      return fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage]);

  const allTrips: PaxMonFilteredTripInfo[] = data
    ? data.pages.flatMap((p) => p.trips)
    : [];
  const totalNumberOfTrips = data?.pages[0]?.total_matching_trips;

  const selectedTripId = params.tripId;

  const scheduleRange = getScheduleRange(scheduleInfo);
  if (selectedDate === undefined && scheduleInfo) {
    setSelectedDate(scheduleRange.closestDate);
  }

  return (
    <div className="flex h-full flex-col">
      <Listbox value={selectedSort} onChange={setSelectedSort}>
        <div className="relative mb-2">
          <Listbox.Button className="relative w-full cursor-default rounded-lg bg-white py-2 pl-3 pr-10 text-left shadow-md focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 dark:bg-gray-700 sm:text-sm">
            <span className="block truncate">{selectedSort.label}</span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon
                className="h-5 w-5 text-gray-400"
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
            <Listbox.Options className="absolute z-20 mt-1 max-h-80 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
              {sortOptions.map((opt) => (
                <Listbox.Option
                  key={opt.option}
                  value={opt}
                  className={({ active }) =>
                    cn(
                      "relative cursor-default select-none py-2 pl-10 pr-4",
                      active ? "bg-amber-100 text-amber-900" : "text-gray-900",
                    )
                  }
                >
                  {({ selected, active }) => (
                    <>
                      <span
                        className={cn(
                          "block truncate",
                          selected ? "font-medium" : "font-normal",
                        )}
                      >
                        {opt.label}
                      </span>
                      {selected ? (
                        <span
                          className={cn(
                            "absolute inset-y-0 left-0 flex items-center pl-3",
                            active ? "text-amber-600" : "text-amber-600",
                          )}
                        >
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
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
      <div className="flex justify-between gap-1 pb-2">
        <div className="">
          <label>
            <span className="text-sm">Datum</span>
            <DatePicker
              value={selectedDate}
              onChange={setSelectedDate}
              min={scheduleRange.firstDay}
              max={scheduleRange.lastDay}
            />
          </label>
        </div>
        <div className="grow">
          <label>
            <span className="text-sm">Zugnummer(n)</span>
            <input
              type="text"
              className="block w-full rounded-md border-gray-300 bg-white text-sm shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 dark:bg-gray-700"
              value={trainNrFilter}
              onChange={(e) => setTrainNrFilter(e.target.value)}
            />
          </label>
        </div>
        <div className="flex flex-col justify-end">
          <ServiceClassFilter
            selectedServiceClasses={serviceClassFilter}
            setSelectedServiceClasses={setServiceClassFilter}
            popupPosition="right-0"
          />
        </div>
      </div>
      {totalNumberOfTrips !== undefined && (
        <div className="flex items-center justify-between">
          <div className="pb-2 text-lg">
            {formatNumber(totalNumberOfTrips)}{" "}
            {totalNumberOfTrips === 1 ? "Zug" : "Züge"}
          </div>
          <div>
            {!isFetching && (
              <button onClick={() => refetch()}>
                <ArrowPathIcon className="h-5 w-5" aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      )}
      <div className="grow">
        {data ? (
          <Virtuoso
            data={allTrips}
            increaseViewportBy={500}
            endReached={loadMore}
            itemContent={(index, ti) => (
              <TripListEntry ti={ti} selectedTripId={selectedTripId} />
            )}
          />
        ) : (
          <div>Züge werden geladen...</div>
        )}
      </div>
    </div>
  );
}

interface TripListEntryProps {
  ti: PaxMonFilteredTripInfo;
  selectedTripId: string | undefined;
}

function TripListEntry({
  ti,
  selectedTripId,
}: TripListEntryProps): JSX.Element {
  const isSelected = selectedTripId === JSON.stringify(ti.tsi.trip);

  const category = ti.tsi.service_infos[0]?.category ?? "";
  const trainNr = ti.tsi.service_infos[0]?.train_nr ?? ti.tsi.trip.train_nr;

  const critSections = ti.edges
    .filter((e) => e.possibly_over_capacity && e.capacity_type === "Known")
    .map((e) => {
      return {
        edge: e,
        maxPax: e.dist.max,
        maxPercent: e.dist.max / e.capacity,
        maxOverCap: Math.max(0, e.dist.max - e.capacity),
      };
    });
  let criticalInfo = null;

  if (critSections.length > 0) {
    const firstCritSection = critSections[0];
    const mostCritSection = critSections.sort(
      (a, b) => b.maxPercent - a.maxPercent,
    )[0];

    criticalInfo = (
      <div className="flex flex-col gap-1 pt-1">
        <SectionOverCap label="Kritisch ab:" section={firstCritSection} />
        {mostCritSection != firstCritSection && (
          <SectionOverCap
            label="Kritischster Abschnitt ab:"
            section={mostCritSection}
          />
        )}
      </div>
    );
  }

  return (
    <div className="pb-3 pr-1">
      <Link
        to={`/trips/${encodeURIComponent(JSON.stringify(ti.tsi.trip))}`}
        className={cn(
          "block rounded p-1",
          isSelected
            ? "bg-db-cool-gray-300 shadow-md dark:bg-gray-500 dark:text-gray-100"
            : "bg-db-cool-gray-100 dark:bg-gray-700 dark:text-gray-300",
        )}
      >
        <div className="flex gap-4 pb-1">
          <div className="flex flex-col">
            <div className="text-center text-sm">{category}</div>
            <div className="text-xl font-semibold">{trainNr}</div>
          </div>
          <div className="flex grow flex-col truncate">
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
          <MiniTripLoadGraph edges={ti.edges} />
        </div>
        {criticalInfo}
      </Link>
    </div>
  );
}

interface SectionOverCapInfo {
  edge: PaxMonEdgeLoadInfo;
  maxPax: number;
  maxPercent: number;
  maxOverCap: number;
}

interface SectionOverCapProps {
  label: string;
  section: SectionOverCapInfo;
}

function SectionOverCap({ label, section }: SectionOverCapProps) {
  return (
    <div>
      <div className="flex justify-between">
        <div className="text-xs">{label}</div>
        <div className="space-x-1 text-xs">
          <span>
            {section.maxOverCap}{" "}
            <abbr title="Reisende über Kapazität" className="no-underline">
              über Kapazität
            </abbr>{" "}
            &ndash;
          </span>
          <span>
            {section.maxPax}/{section.edge.capacity}
          </span>
        </div>
      </div>
      <div className="flex justify-between">
        <div className="space-x-1 truncate">
          <span>{formatTime(section.edge.departure_schedule_time)}</span>
          <span>{section.edge.from.name}</span>
        </div>
        <div className="whitespace-nowrap">
          {formatPercent(section.maxPercent)}
        </div>
      </div>
    </div>
  );
}

export default TripList;
