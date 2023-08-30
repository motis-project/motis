import { ArrowPathIcon } from "@heroicons/react/20/solid";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useAtom } from "jotai";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  ChevronRight,
  Clock,
  CornerDownRight,
  Users,
  XCircle,
} from "lucide-react";
import React, { ReactElement, useCallback } from "react";
import { Link, useParams } from "react-router-dom";
import { Virtuoso } from "react-virtuoso";

import {
  PaxMonBrokenTransfersRequest,
  PaxMonDetailedTransferInfo,
} from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonBrokenTransfersRequest } from "@/api/paxmon";

import { formatShortDuration } from "@/data/durationFormat";
import { universeAtom } from "@/data/multiverse";
import { formatNumber } from "@/data/numberFormat";

import { formatDateTime, formatTime } from "@/util/dateFormat";

import TripServiceInfoView from "@/components/TripServiceInfoView";

import { cn } from "@/lib/utils";

function BrokenTransfersList(): ReactElement {
  const params = useParams();
  const [universe] = useAtom(universeAtom);

  const baseRequest: PaxMonBrokenTransfersRequest = {
    universe,
    filter_interval: { begin: 0, end: 0 },
    ignore_past_transfers: false,
    include_insufficient_transfer_time: true,
    include_missed_initial_departure: true,
    include_canceled_transfer: true,
    include_canceled_initial_departure: true,
    include_canceled_final_arrival: true,
    only_planned_routes: false,
    sort_by: "SquaredTotalDelayIncrease",
    max_results: 100,
    skip_first: 0,
  };

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
  } = useInfiniteQuery(
    queryKeys.brokenTransfers(baseRequest),
    ({ pageParam = 0 }) =>
      sendPaxMonBrokenTransfersRequest({
        ...baseRequest,
        skip_first: pageParam as number,
      }),
    {
      getNextPageParam: (lastPage) =>
        lastPage.remaining_transfers > 0 ? lastPage.next_skip : undefined,
      refetchOnWindowFocus: true,
      keepPreviousData: true,
      staleTime: 60000,
    },
  );

  const loadMore = useCallback(async () => {
    if (hasNextPage) {
      return await fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage]);

  const allTransfers: PaxMonDetailedTransferInfo[] = data
    ? data.pages.flatMap((page) => page.transfers)
    : [];
  const totalNumberOfBrokenTransfers = data?.pages[0]?.total_matching_transfers;

  const selectedTransferN = Number.parseInt(params.n ?? "");
  const selectedTransferE = Number.parseInt(params.e ?? "");

  return (
    <div className="h-full flex flex-col">
      {totalNumberOfBrokenTransfers !== undefined && (
        <div className="flex justify-between items-center">
          <div className="pb-2 text-lg">
            {formatNumber(totalNumberOfBrokenTransfers)}{" "}
            {totalNumberOfBrokenTransfers === 1
              ? "gebrochener Umstieg"
              : "gebrochene Umstiege"}
          </div>
          <div>
            {!isFetching && (
              <button onClick={() => refetch()}>
                <ArrowPathIcon className="w-5 h-5" aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      )}
      <div className="grow">
        {data ? (
          <Virtuoso
            data={allTransfers}
            increaseViewportBy={500}
            endReached={loadMore}
            itemContent={(index, info) => (
              <TransferListEntry
                info={info}
                isSelected={
                  info.id.n === selectedTransferN &&
                  info.id.e === selectedTransferE
                }
              />
            )}
          />
        ) : (
          <div>Umstiege werden geladen...</div>
        )}
      </div>
    </div>
  );
}

interface TransferListEntryProps {
  info: PaxMonDetailedTransferInfo;
  isSelected: boolean;
}

function TransferListEntry({ info, isSelected }: TransferListEntryProps) {
  const arrival = info.arrival[0];
  const departure = info.departure[0];

  const stopInfo = departure ?? arrival;

  if (!stopInfo) {
    // at least one of the two (departure or arrival) is guaranteed to be set, so this should never happen
    return <></>;
  }

  const isBrokenTransfer = arrival && departure;
  const isBrokenDeparture = !arrival && departure;
  const isBrokenArrival = arrival && !departure;
  const differentStations =
    isBrokenTransfer && arrival.station.id !== departure.station.id;

  const availableTransferTime =
    departure && arrival && !info.canceled
      ? (departure.current_time - arrival.current_time) / 60
      : null;

  return (
    <div className="pr-1 pb-3">
      <Link
        to={`/transfers/${info.id.n}/${info.id.e}`}
        className={cn(
          "block p-2 rounded",
          isSelected
            ? "bg-db-cool-gray-300 dark:bg-gray-500 dark:text-gray-100 shadow-md"
            : "bg-db-cool-gray-100 dark:bg-gray-700 dark:text-gray-300",
        )}
      >
        <div className="flex flex-wrap items-center gap-x-1">
          {differentStations ? (
            <>
              {arrival && <div>{arrival.station.name}</div>}
              {differentStations && <ChevronRight className="w-4 h-4" />}
              {departure && <div>{departure.station.name}</div>}
            </>
          ) : (
            <div>{stopInfo.station.name}</div>
          )}
        </div>
        <div className="grid grid-cols-12 items-center mt-2 mb-2">
          {arrival && (
            <>
              <div>
                {isBrokenArrival && <ArrowDownToLine className="w-4 h-4" />}
              </div>
              <div className="col-span-7 flex flex-wrap gap-2">
                {arrival.trips.map((tsi, idx) => (
                  <div key={idx}>
                    <TripServiceInfoView tsi={tsi} format="Short" />
                  </div>
                ))}
              </div>
              <div
                className="col-span-2 text-center"
                title={formatDateTime(arrival.schedule_time)}
              >
                {formatTime(arrival.schedule_time)}
              </div>
              <div className="col-span-2 text-center">
                {arrival.canceled ? (
                  <span className="text-red-600">Ausfall</span>
                ) : (
                  <span title={formatDateTime(arrival.current_time)}>
                    {formatShortDuration(
                      (arrival.current_time - arrival.schedule_time) / 60,
                      true,
                    )}
                  </span>
                )}
              </div>
            </>
          )}
          {departure && (
            <>
              <div>
                {isBrokenTransfer && <CornerDownRight className="w-4 h-4" />}
                {isBrokenDeparture && (
                  <ArrowUpFromLine className="w-4 h-4 rotate-180" />
                )}
              </div>
              <div className="col-span-7 flex flex-wrap gap-2">
                {departure.trips.map((tsi, idx) => (
                  <div key={idx}>
                    <TripServiceInfoView tsi={tsi} format="Short" />
                  </div>
                ))}
              </div>
              <div
                className="col-span-2 text-center"
                title={formatDateTime(departure.schedule_time)}
              >
                {formatTime(departure.schedule_time)}
              </div>
              <div className="col-span-2 text-center">
                {departure.canceled ? (
                  <span className="text-red-600">Ausfall</span>
                ) : (
                  <span title={formatDateTime(departure.current_time)}>
                    {formatShortDuration(
                      (departure.current_time - departure.schedule_time) / 60,
                      true,
                    )}
                  </span>
                )}
              </div>
            </>
          )}
        </div>

        {availableTransferTime !== null && (
          <div className="text-right mb-2">
            Fehlende Umstiegszeit:{" "}
            {formatShortDuration(info.transfer_time - availableTransferTime)}
          </div>
        )}

        <div className="grid grid-cols-12">
          <div
            className="col-span-3 flex items-center gap-x-1"
            title="Betroffene Reisende"
          >
            <Users
              className="w-5 h-5 text-db-cool-gray-500"
              aria-hidden="true"
            />
            {info.pax_count}
            <span className="sr-only">Reisende</span>
          </div>
          <div
            className="col-span-3 flex justify-center items-center gap-x-1"
            title="Gestrandete Reisende"
          >
            {(info.delay?.unreachable_pax ?? 0) > 0 ? (
              <>
                <XCircle
                  className="w-5 h-5 text-db-cool-gray-500"
                  aria-hidden="true"
                />
                <span>{info.delay?.unreachable_pax ?? 0}</span>
                <span className="sr-only">gestrandete Reisende</span>
              </>
            ) : null}
          </div>
          <div
            className="col-span-6 flex justify-end items-center gap-x-1"
            title="Erwartete Zielverspätung der Reisenden"
          >
            {info.delay && info.delay?.unreachable_pax !== info.pax_count && (
              <>
                <Clock
                  className="w-5 h-5 text-db-cool-gray-500"
                  aria-hidden="true"
                />
                <span>
                  {formatShortDuration(info.delay.min_delay_increase)}–
                  {formatShortDuration(info.delay.max_delay_increase)}
                </span>
                <span className="sr-only">Zielverspätung der Reisenden</span>
              </>
            )}
          </div>
        </div>
      </Link>
    </div>
  );
}

export default BrokenTransfersList;
