import { Listbox, Switch, Transition } from "@headlessui/react";
import {
  AdjustmentsVerticalIcon,
  CheckIcon,
  ChevronUpDownIcon,
} from "@heroicons/react/20/solid";
import { MapIcon, UsersIcon, XCircleIcon } from "@heroicons/react/24/outline";
import { useInfiniteQuery } from "@tanstack/react-query";
import { add, fromUnixTime, getUnixTime, max, sub } from "date-fns";
import { useAtom } from "jotai";
import React, { Fragment, useCallback, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Virtuoso } from "react-virtuoso";

import { Station } from "@/api/protocol/motis";
import {
  PaxMonDataSource,
  PaxMonFilterGroupsRequest,
  PaxMonFilterGroupsSortOrder,
  PaxMonGroupRoute,
  PaxMonGroupWithStats,
  PaxMonRerouteReason,
} from "@/api/protocol/motis/paxmon";

import { useLookupScheduleInfoQuery } from "@/api/lookup";
import { sendPaxMonFilterGroupsRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import classNames from "@/util/classNames";
import { formatISODate, formatTime } from "@/util/dateFormat";
import { extractNumbers } from "@/util/extractNumbers";

import StationPicker from "@/components/inputs/StationPicker";
import Delay from "@/components/util/Delay";

type LabeledSortOrder = {
  option: PaxMonFilterGroupsSortOrder;
  label: string;
};

const sortOptions: Array<LabeledSortOrder> = [
  { option: "GroupId", label: "Gruppen sortiert nach ID" },
  {
    option: "ExpectedEstimatedDelay",
    label: "Gruppen sortiert nach erwarteter Verspätung",
  },
  {
    option: "MaxEstimatedDelay",
    label: "Gruppen sortiert nach maximaler Verspätung",
  },
  {
    option: "MinEstimatedDelay",
    label: "Gruppen sortiert nach minimaler Verspätung",
  },
  {
    option: "ScheduledDepartureTime",
    label: "Gruppen sortiert nach Reisebeginn",
  },
  {
    option: "RerouteLogEntries",
    label: "Gruppen sortiert nach Änderungsanzahl",
  },
];

type GroupIdType = "internal" | "source";

type LabeledRerouteReason = {
  reason: PaxMonRerouteReason;
  label: string;
};

const rerouteReasonOptions: Array<LabeledRerouteReason> = [
  { reason: "BrokenTransfer", label: "Gebrochener Umstieg" },
  { reason: "MajorDelayExpected", label: "Hohe erwartete Zielverspätung" },
  { reason: "Simulation", label: "Simulation" },
  { reason: "Manual", label: "Manuelle Umleitung" },
  { reason: "RevertForecast", label: "Rücknahme einer Vorhersage" },
  { reason: "UpdateForecast", label: "Neuberechnung einer Vorhersage" },
  { reason: "DestinationUnreachable", label: "Ziel nicht mehr erreichbar" },
  { reason: "DestinationReachable", label: "Ziel wieder erreichbar" },
];

function getFilterGroupsRequest(
  pageParam: number,
  universe: number,
  sortOrder: PaxMonFilterGroupsSortOrder,
  filterGroupIds: number[],
  filterDataSources: PaxMonDataSource[],
  fromStationFilter: Station | undefined,
  toStationFilter: Station | undefined,
  filterTrainNrs: number[],
  selectedDate: Date | undefined,
  filterByRerouteReason: boolean,
  rerouteReasonFilter: PaxMonRerouteReason[]
): PaxMonFilterGroupsRequest {
  return {
    universe,
    sort_by: sortOrder,
    max_results: 100,
    skip_first: pageParam,
    include_reroute_log: false,
    filter_by_start: fromStationFilter ? [fromStationFilter.id] : [],
    filter_by_destination: toStationFilter ? [toStationFilter.id] : [],
    filter_by_via: [],
    filter_by_group_id: filterGroupIds,
    filter_by_data_source: filterDataSources,
    filter_by_train_nr: filterTrainNrs,
    filter_by_time: selectedDate ? "DepartureOrArrivalTime" : "NoFilter",
    filter_interval: {
      begin: selectedDate ? getUnixTime(selectedDate) : 0,
      end: selectedDate ? getUnixTime(add(selectedDate, { days: 1 })) : 0,
    },
    filter_by_reroute_reason: filterByRerouteReason ? rerouteReasonFilter : [],
  };
}

function GroupList(): JSX.Element {
  const params = useParams();
  const [universe] = useAtom(universeAtom);

  const [selectedSort, setSelectedSort] = useState(sortOptions[0]);
  const [selectedDate, setSelectedDate] = useState<Date>();
  const [fromStationFilter, setFromStationFilter] = useState<
    Station | undefined
  >();
  const [toStationFilter, setToStationFilter] = useState<Station | undefined>();
  const [trainNrFilter, setTrainNrFilter] = useState("");
  const [groupIdFilter, setGroupIdFilter] = useState("");
  const [externalGroupIds, setExternalGroupIds] = useState(false);
  const [filterByRerouteReason, setFilterByRerouteReason] = useState(false);
  const [rerouteReasonFilter, setRerouteReasonFilter] = useState<
    PaxMonRerouteReason[]
  >(rerouteReasonOptions.map((r) => r.reason));

  const filterTrainNrs = extractNumbers(trainNrFilter);

  const filterGroupIdsInput = extractNumbers(groupIdFilter);
  const idType: GroupIdType = externalGroupIds ? "source" : "internal";
  const filterGroupIds: number[] =
    idType == "internal" ? filterGroupIdsInput : [];
  const filterDataSources: PaxMonDataSource[] =
    idType == "source"
      ? filterGroupIdsInput.map((id) => {
          return { primary_ref: id, secondary_ref: 0 };
        })
      : [];

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
      "groupList",
      {
        universe,
        sortOrder: selectedSort.option,
        filterGroupIds,
        filterDataSources,
        fromStationFilter,
        toStationFilter,
        filterTrainNrs,
        selectedDate,
        rerouteReasonFilter: filterByRerouteReason ? rerouteReasonFilter : [],
      },
    ],
    ({ pageParam = 0 }) => {
      const req = getFilterGroupsRequest(
        pageParam,
        universe,
        selectedSort.option,
        filterGroupIds,
        filterDataSources,
        fromStationFilter,
        toStationFilter,
        filterTrainNrs,
        selectedDate,
        filterByRerouteReason,
        rerouteReasonFilter
      );
      return sendPaxMonFilterGroupsRequest(req);
    },
    {
      getNextPageParam: (lastPage) =>
        lastPage.remaining_groups > 0 ? lastPage.next_skip : undefined,
      refetchOnWindowFocus: true,
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

  const allGroups: PaxMonGroupWithStats[] = data
    ? data.pages.flatMap((p) => p.groups)
    : [];
  const totalNumberOfGroups = data?.pages[0]?.total_matching_groups;
  const totalNumberOfPassengers = data?.pages[0]?.total_matching_passengers;

  const selectedGroup = Number.parseInt(params["groupId"] ?? "");

  return (
    <div className="h-full flex flex-col">
      <Listbox value={selectedSort} onChange={setSelectedSort}>
        <div className="relative mb-2">
          <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white dark:bg-gray-700 rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
            <span className="block truncate">{selectedSort.label}</span>
            <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <ChevronUpDownIcon
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
      <label>
        <span className="text-sm">Startstation</span>
        <StationPicker
          onStationPicked={setFromStationFilter}
          clearOnPick={false}
          clearButton={true}
          initialStation={fromStationFilter}
        />
      </label>
      <label>
        <span className="text-sm">Zielstation</span>
        <StationPicker
          onStationPicked={setToStationFilter}
          clearOnPick={false}
          clearButton={true}
          initialStation={toStationFilter}
        />
      </label>
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
        <div className="grow">
          <label>
            <span className="text-sm">Verwendete Zugnummer(n)</span>
            <input
              type="text"
              className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              value={trainNrFilter}
              onChange={(e) => setTrainNrFilter(e.target.value)}
            />
          </label>
        </div>
      </div>
      <div className="flex justify-between pb-2 gap-1">
        <div className="grow">
          <div className="flex justify-between">
            <div className="text-sm">Gruppen-ID(s)</div>
            <div className="flex items-center gap-1">
              <span className="text-xs">Interne IDs</span>
              <Switch
                checked={externalGroupIds}
                onChange={setExternalGroupIds}
                className={`bg-db-cool-gray-500 relative inline-flex h-[18px] w-[38px]
                  shrink-0 cursor-pointer rounded-full border-2 border-transparent
                  duration-200 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75`}
              >
                <span
                  aria-hidden="true"
                  className={`ui-not-checked:translate-x-0 ui-checked:translate-x-5
                  pointer-events-none inline-block h-[14px] w-[14px] transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out`}
                />
              </Switch>
              <span className="text-xs">Quell-IDs</span>
            </div>
          </div>
          <input
            type="text"
            className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            value={groupIdFilter}
            onChange={(e) => setGroupIdFilter(e.target.value)}
          />
        </div>
        <div className="flex items-start"></div>
      </div>
      <div className="flex justify-between pb-2 gap-1">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="rounded border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-offset-0 focus:ring-blue-200 focus:ring-opacity-50"
            checked={filterByRerouteReason}
            onChange={() => setFilterByRerouteReason((c) => !c)}
          />
          Nur umgeleitete Gruppen
        </label>
        <div className="flex flex-col justify-end">
          <RerouteReasonOptions
            rerouteReasonFilter={rerouteReasonFilter}
            setRerouteReasonFilter={setRerouteReasonFilter}
          />
        </div>
      </div>
      {totalNumberOfGroups !== undefined && (
        <div className="pb-2 text-lg">
          {formatNumber(totalNumberOfGroups)}{" "}
          {totalNumberOfGroups === 1 ? "Gruppe" : "Gruppen"}
          {` (${formatNumber(totalNumberOfPassengers || 0)} Reisende)`}
        </div>
      )}
      <div className="grow">
        {data ? (
          <Virtuoso
            data={allGroups}
            increaseViewportBy={500}
            endReached={loadMore}
            itemContent={(index, groupWithStats) => (
              <GroupListEntry
                groupWithStats={groupWithStats}
                idType={idType}
                selectedGroup={selectedGroup}
              />
            )}
          />
        ) : (
          <div>Gruppen werden geladen...</div>
        )}
      </div>
    </div>
  );
}

type RerouteReasonOptionsProps = {
  rerouteReasonFilter: PaxMonRerouteReason[];
  setRerouteReasonFilter: React.Dispatch<
    React.SetStateAction<PaxMonRerouteReason[]>
  >;
};

function RerouteReasonOptions({
  rerouteReasonFilter,
  setRerouteReasonFilter,
}: RerouteReasonOptionsProps) {
  return (
    <Listbox
      value={rerouteReasonFilter}
      onChange={setRerouteReasonFilter}
      multiple
    >
      <div className="relative">
        <Listbox.Button className="p-2 mb-0.5 flex justify-center align-center bg-white text-black dark:bg-gray-600 dark:text-gray-100 rounded-full shadow-sm outline-0">
          <AdjustmentsVerticalIcon className="w-5 h-5" aria-hidden="true" />
        </Listbox.Button>
        <Transition
          as={Fragment}
          leave="transition ease-in duration-100"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <Listbox.Options className="absolute right-0 z-20 py-1 mt-1 overflow-auto text-base bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {rerouteReasonOptions.map((opt) => (
              <Listbox.Option
                key={opt.reason}
                value={opt.reason}
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
  );
}

type GroupRouteInfoProps = {
  route: PaxMonGroupRoute;
};

function GroupRouteInfo({ route }: GroupRouteInfoProps): JSX.Element {
  const firstLeg = route.journey.legs[0];
  const lastLeg = route.journey.legs[route.journey.legs.length - 1];
  return (
    <div className="flex flex-col truncate pl-2">
      <div className="flex justify-between">
        <div className="truncate">{firstLeg.enter_station.name}</div>
        <div>{formatTime(firstLeg.enter_time)}</div>
      </div>
      <div className="flex justify-between">
        <div className="truncate">{lastLeg.exit_station.name}</div>
        <div>{formatTime(lastLeg.exit_time)}</div>
      </div>
    </div>
  );
}

type GroupListEntryProps = {
  groupWithStats: PaxMonGroupWithStats;
  idType: GroupIdType;
  selectedGroup: number | undefined;
};

function GroupListEntry({
  groupWithStats,
  idType,
  selectedGroup,
}: GroupListEntryProps): JSX.Element {
  const group = groupWithStats.group;
  const totalRouteCount = group.routes.length;
  const activeRouteCount = group.routes.filter((r) => r.probability > 0).length;
  const firstRoute = totalRouteCount > 0 ? group.routes[0] : null;
  const isSelected = selectedGroup === group.id;

  return (
    <div className="pr-1 pb-3">
      <Link
        to={`/groups/${group.id}`}
        className={classNames(
          "block p-2 rounded",
          isSelected
            ? "bg-db-cool-gray-300 dark:bg-gray-500 dark:text-gray-100 shadow-md"
            : "bg-db-cool-gray-100 dark:bg-gray-700 dark:text-gray-300"
        )}
      >
        <div className="flex justify-between">
          {idType === "internal" ? (
            <div
              title={`Quelle: ${group.source.primary_ref}.${group.source.secondary_ref}`}
            >
              Gruppe {group.id}
            </div>
          ) : (
            <div title={`Interne ID: ${group.id}`}>
              Gruppe <span>{group.source.primary_ref}</span>
              <span className="text-db-cool-gray-400">
                .{group.source.secondary_ref}
              </span>
            </div>
          )}
          <div className="flex items-center gap-x-1">
            <UsersIcon
              className="w-5 h-5 text-db-cool-gray-500"
              aria-hidden="true"
            />
            {group.passenger_count}
            <span className="sr-only">Reisende</span>
          </div>
        </div>
        {firstRoute && <GroupRouteInfo route={firstRoute} />}
        <div className="flex justify-between">
          {groupWithStats.prob_destination_unreachable == 0 ? (
            <div className="flex gap-1">
              Verspätung:
              <Delay minutes={groupWithStats.expected_estimated_delay} />
              {groupWithStats.min_estimated_delay !=
              groupWithStats.max_estimated_delay ? (
                <div>
                  (<Delay minutes={groupWithStats.min_estimated_delay} /> –{" "}
                  <Delay minutes={groupWithStats.max_estimated_delay} />)
                </div>
              ) : null}
            </div>
          ) : (
            <div className="flex items-center gap-x-1 text-fuchsia-600">
              <XCircleIcon className="w-5 h-5" aria-hidden="true" />
              Ziel nicht erreichbar (
              {formatPercent(groupWithStats.prob_destination_unreachable)})
            </div>
          )}
          <div className="flex items-center gap-x-1">
            <MapIcon
              className="w-5 h-5 text-db-cool-gray-500"
              aria-hidden="true"
            />
            {activeRouteCount}/{totalRouteCount}
            <span className="sr-only">Routen</span>
          </div>
        </div>
      </Link>
    </div>
  );
}

export default GroupList;
