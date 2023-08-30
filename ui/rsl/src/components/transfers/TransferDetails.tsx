import { useQuery } from "@tanstack/react-query";
import { useAtom } from "jotai/index";
import { ArrowRight, ChevronRight, Clock, Users, XCircle } from "lucide-react";
import React, { ReactNode } from "react";
import { Link, useParams } from "react-router-dom";

import { TripServiceInfo } from "@/api/protocol/motis";
import {
  PaxMonGroup,
  PaxMonTransferDetailsRequest,
  PaxMonTransferId,
} from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonTransferDetailsRequest } from "@/api/paxmon";

import { formatShortDuration } from "@/data/durationFormat";
import { universeAtom } from "@/data/multiverse";

import { formatDateTime } from "@/util/dateFormat";
import {
  getDestinationStation,
  getTotalPaxCount,
  groupHasActiveUnreachableRoutes,
} from "@/util/groups";

import TripServiceInfoView from "@/components/TripServiceInfoView";

import { cn } from "@/lib/utils";

interface TransferDetailsProps {
  transferId: PaxMonTransferId;
}

function TransferDetails({ transferId }: TransferDetailsProps): ReactNode {
  const [universe] = useAtom(universeAtom);

  const req: PaxMonTransferDetailsRequest = {
    universe,
    id: transferId,
    include_disabled_group_routes: true,
    include_full_groups: true,
    include_reroute_log: false,
  };
  const { data, isLoading, error } = useQuery(
    queryKeys.transferDetails(req),
    () => sendPaxMonTransferDetailsRequest(req),
  );

  if (!data) {
    if (isLoading) {
      return <div>Umstiegsinformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Umstiegsinformationen:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
        </div>
      );
    }
  }

  const arrival = data.info.arrival[0];
  const departure = data.info.departure[0];

  const stopInfo = departure ?? arrival;

  if (!stopInfo) {
    // at least one of the two (departure or arrival) is guaranteed to be set, so this should never happen
    return <></>;
  }

  const isBrokenTransfer = arrival && departure;
  const isBrokenDeparture = !arrival && departure;
  const differentStations =
    isBrokenTransfer && arrival.station.id !== departure.station.id;

  const availableTransferTime =
    departure && arrival && !data.info.canceled
      ? (departure.current_time - arrival.current_time) / 60
      : null;

  const groupedByDestination = data.groups.reduce(
    (result, group) => {
      const destinationStationId = getDestinationStation(group).id;

      if (!result[destinationStationId]) {
        result[destinationStationId] = [];
      }

      result[destinationStationId].push(group);

      return result;
    },
    {} as Record<string, PaxMonGroup[]>,
  );

  const sortedGroupedByDestination = Object.entries(groupedByDestination).sort(
    (a, b) => getTotalPaxCount(b[1]) - getTotalPaxCount(a[1]),
  );

  return (
    <div>
      <h1 className="flex flex-wrap items-center gap-2 text-xl">
        {isBrokenTransfer
          ? "Umstieg in"
          : isBrokenDeparture
          ? "Abfahrt in"
          : "Ankunft in"}
        {differentStations ? (
          <>
            {arrival && <span>{arrival.station.name}</span>}
            {differentStations && <ChevronRight className="w-5 h-5" />}
            {departure && <span>{departure.station.name}</span>}
          </>
        ) : (
          <span>{stopInfo.station.name}</span>
        )}
      </h1>

      <div className="max-w-5xl grid grid-cols-9 items-start gap-1 mt-2 mb-2">
        {arrival && (
          <>
            <div>Ankunft mit:</div>
            <div className="flex flex-col gap-1">
              {arrival.trips.map((tsi, idx) => (
                <div key={idx}>
                  <TripServiceInfoView tsi={tsi} format="Short" link={true} />
                </div>
              ))}
            </div>
            <div className="col-span-4">
              {arrival.trips.map((tsi, idx) => (
                <TripRoute key={idx} tsi={tsi} />
              ))}
            </div>
            <div className="col-span-2 text-center">
              {formatDateTime(arrival.schedule_time)}
            </div>
            <div className="text-center">
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
            <div>Abfahrt mit:</div>
            <div className="flex flex-wrap gap-2">
              {departure.trips.map((tsi, idx) => (
                <div key={idx}>
                  <TripServiceInfoView tsi={tsi} format="Short" link={true} />
                </div>
              ))}
            </div>
            <div className="col-span-4">
              {departure.trips.map((tsi, idx) => (
                <TripRoute key={idx} tsi={tsi} />
              ))}
            </div>
            <div className="col-span-2 text-center">
              {formatDateTime(departure.schedule_time)}
            </div>
            <div className="text-center">
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
        <div className="mb-2">
          Fehlende Umstiegszeit:{" "}
          {formatShortDuration(data.info.transfer_time - availableTransferTime)}
        </div>
      )}

      <div className="flex items-center gap-1 mb-2 mt-4">
        <Users className="w-4 h-4" aria-hidden="true" />
        {`${data.info.pax_count} betroffene Reisende`}
      </div>

      {(data.info.delay?.unreachable_pax ?? 0) > 0 ? (
        <div className="flex items-center gap-1 mb-2">
          <XCircle className="w-4 h-4" aria-hidden="true" />
          {`${data.info.delay?.unreachable_pax ?? 0} gestrandete Reisende`}
        </div>
      ) : null}

      {data.info.delay &&
        data.info.delay?.unreachable_pax !== data.info.pax_count && (
          <div className="flex items-center gap-1 mb-2">
            <Clock className="w-4 h-4" aria-hidden="true" />
            <span>
              {formatShortDuration(data.info.delay.min_delay_increase)}–
              {formatShortDuration(data.info.delay.max_delay_increase)}
            </span>
            <span>erwartete Zielverspätung der Reisenden</span>
          </div>
        )}

      <h2 className="text-lg mt-6 mb-2">Betroffene Reisende nach Ziel</h2>
      <div className="flex flex-col gap-1">
        {sortedGroupedByDestination.map(([stationId, groups]) => (
          <GroupsByStation groups={groups} key={stationId} />
        ))}
      </div>

      <h2 className="text-lg mt-6 mb-2">Betroffene Reisendengruppen</h2>
      <div className="flex flex-wrap gap-1">
        {data.groups.map((g) => (
          <Link
            key={g.id}
            to={`/groups/${g.id}`}
            className={cn(
              "w-24 px-2 py-1 rounded text-center",
              groupHasActiveUnreachableRoutes(g) ? "bg-red-200" : "bg-gray-200",
            )}
          >
            {g.id}
          </Link>
        ))}
      </div>
    </div>
  );
}

interface TripRouteProps {
  tsi: TripServiceInfo;
}

function TripRoute({ tsi }: TripRouteProps) {
  return (
    <div className="flex items-center gap-1">
      <div>{tsi.primary_station.name}</div>
      <div>
        <ArrowRight className="w-4 h-4" aria-hidden="true" />
        <span className="sr-only">nach</span>
      </div>
      <div>{tsi.secondary_station.name}</div>
    </div>
  );
}

interface GroupsByStationProps {
  groups: PaxMonGroup[];
}

function GroupsByStation({ groups }: GroupsByStationProps) {
  const station = getDestinationStation(groups[0]);
  const totalPax = getTotalPaxCount(groups);

  return (
    <div className="flex items-center gap-1">
      <Users className="w-4 h-4" aria-hidden="true" />
      <div className="w-10">
        {totalPax} <span className="sr-only">Reisende</span>
      </div>
      <ArrowRight className="w-4 h-4" aria-hidden="true" />
      <div>
        <span className="sr-only">mit Ziel </span>
        {station.name}
      </div>
    </div>
  );
}

export function TransferDetailsFromRoute(): ReactNode {
  const params = useParams();
  const n = Number.parseInt(params.n ?? "");
  const e = Number.parseInt(params.e ?? "");
  if (!Number.isNaN(n) && !Number.isNaN(e)) {
    return <TransferDetails transferId={{ n, e }} />;
  } else {
    return null;
  }
}

export default TransferDetails;
