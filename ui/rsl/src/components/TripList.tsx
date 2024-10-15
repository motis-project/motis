import { Listbox, Transition } from "@headlessui/react";
import {
  AdjustmentsIcon,
  CheckIcon,
  SelectorIcon,
} from "@heroicons/react/solid";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { add, fromUnixTime, getUnixTime, max, sub } from "date-fns";
import { useAtom } from "jotai";
import React, { Fragment, useCallback, useState } from "react";
import { useInfiniteQuery } from "react-query";
import { Virtuoso } from "react-virtuoso";

import { TripServiceInfo } from "@/api/protocol/motis";
import {
  PaxMonEdgeLoadInfo,
  PaxMonFilterTripsRequest,
  PaxMonFilterTripsSortOrder,
  PaxMonFilteredTripInfo,
} from "@/api/protocol/motis/paxmon";

import { ServiceClass } from "@/api/constants";
import { useLookupScheduleInfoQuery } from "@/api/lookup";
import { sendPaxMonFilterTripsRequest } from "@/api/paxmon";

import { formatPercent } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatISODate, formatTime } from "@/util/dateFormat";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";

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
  { option: "MaxPaxRange", label: "Züge sortiert nach Unsicherheit" },
];

function getFilterTripsRequest(
  universe: number,
  sortOrder: PaxMonFilterTripsSortOrder,
  selectedDate: Date | undefined,
  filterTrainNrs: number[],
  pageParam: number,
  serviceClassFilter: number[]
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
    filter_by_service_class: true,
    filter_service_classes: serviceClassFilter,
  };
}

function TripList(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);

  const [selectedSort, setSelectedSort] = useState(sortOptions[0]);
  const [selectedDate, setSelectedDate] = useState<Date>();
  const [trainNrFilter, setTrainNrFilter] = useState("");
  const [serviceClassFilter, setServiceClassFilter] = useState([
    ServiceClass.ICE,
    ServiceClass.IC,
  ]);

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
        serviceClassFilter,
      },
    ],
    ({ pageParam = 0 }) => {
      const req = getFilterTripsRequest(
        universe,
        selectedSort.option,
        selectedDate,
        filterTrainNrs,
        pageParam,
        serviceClassFilter
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
          <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white dark:bg-gray-700 rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
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
      <div className="flex justify-between pb-2 gap-1">
        <div className="">
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
              className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
          </label>
        </div>
        <div className="">
          <label>
            <span className="text-sm">Zugnummer(n)</span>
            <input
              type="text"
              className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              value={trainNrFilter}
              onChange={(e) => setTrainNrFilter(e.target.value)}
            />
          </label>
        </div>
        <div className="flex flex-col justify-end">
          <TripListOptions
            serviceClassFilter={serviceClassFilter}
            setServiceClassFilter={setServiceClassFilter}
          />
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

type LabeledServiceClass = {
  sc: ServiceClass;
  label: string;
};

const serviceClassOptions: Array<LabeledServiceClass> = [
  {
    sc: ServiceClass.ICE,
    label: "Hochgeschwindigkeitszüge",
  },
  { sc: ServiceClass.IC, label: "Fernzüge" },
  { sc: ServiceClass.COACH, label: "Fernbusse" },
  { sc: ServiceClass.N, label: "Nachtzüge" },
  { sc: ServiceClass.RE, label: "Regional-Express" },
  { sc: ServiceClass.RB, label: "Regionalbahnen" },
  { sc: ServiceClass.S, label: "S-Bahnen" },
  { sc: ServiceClass.U, label: "U-Bahnen" },
  { sc: ServiceClass.STR, label: "Straßenbahnen" },
  { sc: ServiceClass.BUS, label: "Busse" },
  { sc: ServiceClass.SHIP, label: "Schiffe" },
  { sc: ServiceClass.AIR, label: "Flugzeuge" },
  { sc: ServiceClass.OTHER, label: "Sonstige" },
];

type TripListOptionsProps = {
  serviceClassFilter: number[];
  setServiceClassFilter: React.Dispatch<React.SetStateAction<number[]>>;
};

function TripListOptions({
  serviceClassFilter,
  setServiceClassFilter,
}: TripListOptionsProps) {
  const toggleClass = useCallback(
    (sc: ServiceClass, checked: boolean) => {
      if (checked) {
        setServiceClassFilter((classes) => [...classes, sc]);
      } else {
        setServiceClassFilter((classes) => classes.filter((c) => c != sc));
      }
    },
    [setServiceClassFilter]
  );

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild={true}>
        <button
          type="button"
          className="p-2 mb-1 flex justify-center align-center bg-white text-black dark:bg-gray-600 dark:text-gray-100 rounded-full shadow-sm outline-0"
        >
          <AdjustmentsIcon className="w-4 h-4" aria-hidden="true" />
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content
        sideOffset={5}
        className="bg-white rounded-md shadow-lg py-1 px-1"
      >
        {serviceClassOptions.map((opt) => (
          <DropdownMenu.CheckboxItem
            key={opt.sc}
            checked={serviceClassFilter.includes(opt.sc)}
            onCheckedChange={(b) => toggleClass(opt.sc, b)}
            className="relative py-2 pl-10 pr-4 text-gray-900 select-none focus:text-amber-900 focus:bg-amber-100 outline-0 rounded-md"
          >
            <DropdownMenu.ItemIndicator asChild={true}>
              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-amber-600">
                <CheckIcon className="w-5 h-5" aria-hidden="true" />
              </span>
            </DropdownMenu.ItemIndicator>
            {opt.label}
          </DropdownMenu.CheckboxItem>
        ))}

        <DropdownMenu.Arrow className="fill-white" />
      </DropdownMenu.Content>
    </DropdownMenu.Root>
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
      (a, b) => b.maxPercent - a.maxPercent
    )[0];

    criticalInfo = (
      <div className="pt-1 flex flex-col gap-1">
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
    <div className="pr-1 pb-3">
      <div
        className={classNames(
          "cursor-pointer p-1 rounded",
          isSelected
            ? "bg-db-cool-gray-300 dark:bg-gray-500 dark:text-gray-100 shadow-md"
            : "bg-db-cool-gray-100 dark:bg-gray-700 dark:text-gray-300"
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
          <MiniTripLoadGraph edges={ti.edges} />
        </div>
        {criticalInfo}
      </div>
    </div>
  );
}

type SectionOverCapInfo = {
  edge: PaxMonEdgeLoadInfo;
  maxPax: number;
  maxPercent: number;
  maxOverCap: number;
};

type SectionOverCapProps = {
  label: string;
  section: SectionOverCapInfo;
};

function SectionOverCap({ label, section }: SectionOverCapProps) {
  return (
    <div>
      <div className="flex justify-between">
        <div className="text-xs">{label}</div>
        <div className="text-xs space-x-1">
          <span>
            {section.maxPax}/{section.edge.capacity}
          </span>
          <span>
            ({section.maxOverCap}{" "}
            <abbr title="Reisende über Kapazität" className="no-underline">
              ü.K.
            </abbr>
            )
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
