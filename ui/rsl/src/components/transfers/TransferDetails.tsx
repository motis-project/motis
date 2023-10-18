import { useQuery } from "@tanstack/react-query";
import { useAtom } from "jotai/index";
import { ArrowRight, ChevronRight, Clock, Users, XCircle } from "lucide-react";
import React, { ReactNode } from "react";
import { Link, useParams } from "react-router-dom";

import { Station, TripServiceInfo } from "@/api/protocol/motis";
import {
  PaxMonGroup,
  PaxMonTransferDetailsRequest,
  PaxMonTransferId,
} from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonTransferDetailsRequest } from "@/api/paxmon";

import { formatShortDuration } from "@/data/durationFormat";
import { universeAtom } from "@/data/multiverse";
import { formatPercent } from "@/data/numberFormat";

import { formatDateTime } from "@/util/dateFormat";
import {
  getDestinationStation,
  getJourneyLegAfterTransfer,
  getTotalPaxCount,
  groupHasActiveUnreachableRoutes,
} from "@/util/groups";

import TripServiceInfoView from "@/components/TripServiceInfoView";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import Delay from "@/components/util/Delay";

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
  const { data, isPending, error } = useQuery({
    queryKey: queryKeys.transferDetails(req),
    queryFn: () => sendPaxMonTransferDetailsRequest(req),
  });

  if (!data) {
    if (isPending) {
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
          <GroupButton
            group={g}
            key={g.id}
            transferArrivalStation={arrival?.station}
            transferDepartureStation={departure?.station}
          />
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

  const maybeUnreachable = groups.some((g) =>
    g.routes.some((r) => r.destination_unreachable && r.probability !== 0),
  );

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
      {maybeUnreachable && (
        <>
          <XCircle
            className="w-4 h-4 text-db-cool-gray-500 ml-4"
            aria-hidden="true"
          />
          <div className="text-red-600">
            Ziel möglicherweise nicht erreichbar
          </div>
        </>
      )}
    </div>
  );
}

interface GroupButtonProps {
  group: PaxMonGroup;
  transferArrivalStation: Station | undefined;
  transferDepartureStation: Station | undefined;
}

function GroupButton({
  group,
  transferArrivalStation,
  transferDepartureStation,
}: GroupButtonProps) {
  const plannedRoute = group.routes[0];
  const plannedDeparture = plannedRoute.journey.legs[0];
  const plannedArrival =
    plannedRoute.journey.legs[plannedRoute.journey.legs.length - 1];

  const activeRoutes = group.routes
    .filter((r) => r.probability > 0)
    .sort((a, b) => b.probability - a.probability)
    .map((r) => {
      return {
        ...r,
        alternativeLeg: getJourneyLegAfterTransfer(
          r.journey,
          transferArrivalStation,
          transferDepartureStation,
        ),
      };
    });

  return (
    <HoverCard openDelay={200}>
      <HoverCardTrigger asChild>
        <Link
          to={`/groups/${group.id}`}
          className={cn(
            "w-24 px-2 py-1 rounded text-center",
            groupHasActiveUnreachableRoutes(group)
              ? "bg-red-200"
              : "bg-gray-200",
          )}
        >
          {group.id}
        </Link>
      </HoverCardTrigger>
      <HoverCardContent className="w-96">
        <div className="flex justify-between gap-1">
          <div className="font-semibold">Planmäßige Verbindung:</div>
          <div className="flex justify-end items-center gap-1">
            <Users className="w-4 h-4" aria-hidden="true" />
            {group.passenger_count}
          </div>
        </div>
        <div className="flex gap-2 items-center">
          <div className="w-5"></div>
          <div>{formatDateTime(plannedDeparture.enter_time)}</div>
          <div>{plannedDeparture.enter_station.name}</div>
        </div>
        <div className="flex gap-2 items-center">
          <div className="w-5">
            <ArrowRight className="w-4 h-4" aria-hidden="true" />
          </div>
          <div>{formatDateTime(plannedArrival.exit_time)}</div>
          <div>{plannedArrival.exit_station.name}</div>
        </div>
        <div className="font-semibold mt-4">
          Zielverspätung mit aktuellen Alternativen:
        </div>
        <table>
          <tbody>
            {activeRoutes.map((route) => (
              <tr key={route.index}>
                <td className="pr-4">{formatPercent(route.probability)}</td>
                <td className="pr-4">
                  {route.broken ? (
                    <span className="text-red-600">Ziel nicht erreichbar</span>
                  ) : (
                    <Delay minutes={route.estimated_delay} forceSign={true} />
                  )}
                </td>
                <td>
                  {route.alternativeLeg && (
                    <>
                      via{" "}
                      <TripServiceInfoView
                        tsi={route.alternativeLeg.trip}
                        format="Short"
                      />
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </HoverCardContent>
    </HoverCard>
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
