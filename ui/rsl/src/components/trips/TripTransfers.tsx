import { useQuery } from "@tanstack/react-query";
import { useAtom } from "jotai/index";
import { zip } from "lodash-es";
import {
  ArrowRightFromLine,
  ArrowRightToLine,
  TrainFront,
  Users,
  Watch,
  XCircle,
} from "lucide-react";
import React, { ReactNode } from "react";
import { Link } from "react-router-dom";

import { TripId } from "@/api/protocol/motis.ts";
import {
  PaxMonDetailedTransferInfo,
  PaxMonTripTransfersAtStop,
  PaxMonTripTransfersRequest,
} from "@/api/protocol/motis/paxmon.ts";

import { queryKeys, sendPaxMonTripTransfersRequest } from "@/api/paxmon.ts";

import { universeAtom } from "@/data/multiverse.ts";

import { formatTime } from "@/util/dateFormat.ts";

import TripServiceInfoView from "@/components/TripServiceInfoView.tsx";

import { cn } from "@/lib/utils.ts";

interface TripTransfersProps {
  tripId: TripId;
}

export function TripTransfers({ tripId }: TripTransfersProps): ReactNode {
  const [universe] = useAtom(universeAtom);

  const request: PaxMonTripTransfersRequest = {
    universe,
    trip: tripId,
    include_delay_info: true,
  };
  const { data, isPending, error } = useQuery({
    queryKey: queryKeys.tripTransfers(request),
    queryFn: () => sendPaxMonTripTransfersRequest(request),
  });

  if (!data) {
    if (isPending) {
      return <div>Anschlussinformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Anschlussinformationen:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
        </div>
      );
    }
  }

  return (
    <div>
      {zip(
        [...data.incoming_transfers, undefined],
        [undefined, ...data.outgoing_transfers],
      ).map(([incoming, outgoing], idx) => (
        <TransfersAtStop incoming={incoming} key={idx} outgoing={outgoing} />
      ))}
    </div>
  );
}

interface TransfersAtStopProps {
  incoming: PaxMonTripTransfersAtStop | undefined;
  outgoing: PaxMonTripTransfersAtStop | undefined;
}

function TransfersAtStop({ incoming, outgoing }: TransfersAtStopProps) {
  const station = incoming?.station ?? outgoing?.station;

  if (!station) {
    return null;
  }

  return (
    <div className="my-12">
      <div className="bg-gray-100 text-center text-xl">{station.name}</div>
      <div className="flex">
        <div className="w-1/2">
          {incoming && (
            <TransferList transfers={incoming} direction="incoming" />
          )}
        </div>
        <div className="w-1/2">
          {outgoing && (
            <TransferList transfers={outgoing} direction="outgoing" />
          )}
        </div>
      </div>
    </div>
  );
}

interface TransferListProps {
  transfers: PaxMonTripTransfersAtStop;
  direction: "incoming" | "outgoing";
}

function TransferList({ transfers, direction }: TransferListProps) {
  const thisTrip =
    direction == "incoming"
      ? transfers.transfers[0]?.departure[0]
      : transfers.transfers[0]?.arrival[0];

  const getStopInfo = (dti: PaxMonDetailedTransferInfo) =>
    direction == "incoming" ? dti.arrival[0] : dti.departure[0];
  const sortedTransfers = [...transfers.transfers];
  sortedTransfers.sort(
    (a, b) =>
      (getStopInfo(a)?.schedule_time ?? 0) -
      (getStopInfo(b)?.schedule_time ?? 0),
  );

  return (
    <div className="flex flex-col gap-1">
      <div className="my-2 flex gap-6 font-semibold">
        {thisTrip && (
          <div className="flex gap-2">
            <div>{direction == "incoming" ? "Abfahrt:" : "Ankunft:"}</div>
            <div className="flex w-12 items-center gap-1">
              {formatTime(thisTrip.schedule_time)}
            </div>
            <div
              className={cn(
                "flex w-12 items-center gap-1",
                thisTrip.current_time > thisTrip.schedule_time
                  ? "text-red-600"
                  : "text-green-600",
              )}
            >
              {formatTime(thisTrip.current_time)}
            </div>
          </div>
        )}
        <div>
          {transfers.transfers.length}{" "}
          {direction == "incoming" ? "eingehende" : "ausgehende"} Anschlüsse
        </div>
      </div>
      {sortedTransfers.map((dti) => (
        <TransferEntry
          transfer={dti}
          direction={direction}
          key={`${dti.id.n}.${dti.id.e}`}
        />
      ))}
    </div>
  );
}

interface TransferEntryProps {
  transfer: PaxMonDetailedTransferInfo;
  direction: "incoming" | "outgoing";
}

function TransferEntry({ transfer, direction }: TransferEntryProps) {
  const stopInfo =
    direction == "incoming" ? transfer.arrival[0] : transfer.departure[0];
  const thisTrip =
    direction == "incoming" ? transfer.departure[0] : transfer.arrival[0];

  const availableTransferTime =
    stopInfo && thisTrip
      ? direction == "incoming"
        ? (thisTrip.current_time - stopInfo.current_time) / 60
        : (stopInfo.current_time - thisTrip.current_time) / 60
      : 0;

  return (
    <div className="flex gap-1">
      <div className="flex w-6 items-center pl-1">
        {transfer.broken && (
          <XCircle className="h-4 w-4 text-red-600" aria-hidden="true" />
        )}
      </div>
      <Link to={`/transfers/${transfer.id.n}/${transfer.id.e}`}>
        <div className="flex w-16 items-center gap-1">
          <Users className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <span>{transfer.pax_count}</span>
        </div>
      </Link>
      <div className="flex w-32 items-center gap-1">
        {stopInfo ? (
          <>
            <Link
              to={`/trips/${encodeURIComponent(JSON.stringify(stopInfo.trips[0].trip))}`}
              className="flex items-center gap-1"
            >
              <TrainFront
                className="h-4 w-4 text-muted-foreground"
                aria-hidden="true"
              />
              <TripServiceInfoView tsi={stopInfo.trips[0]} format="Short" />
            </Link>
          </>
        ) : direction == "incoming" ? (
          <>
            <ArrowRightFromLine
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
            <span>Reisebeginn</span>
          </>
        ) : (
          <>
            <ArrowRightToLine
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
            <span>Reiseziel</span>
          </>
        )}
      </div>
      {stopInfo && (
        <>
          <div className="flex w-14 items-center gap-1">
            {formatTime(stopInfo.schedule_time)}
          </div>
          <div
            className={cn(
              "flex w-14 items-center gap-1",
              stopInfo.current_time > stopInfo.schedule_time ||
                stopInfo.canceled
                ? "text-red-600"
                : "text-green-600",
            )}
          >
            {stopInfo.canceled ? "Ausfall" : formatTime(stopInfo.current_time)}
          </div>
          <div className="flex w-24 items-center gap-1">
            <Watch
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
            <span
              className={cn(
                availableTransferTime >= transfer.transfer_time &&
                  !stopInfo.canceled
                  ? "text-green-600"
                  : "text-red-600",
              )}
            >
              {stopInfo.canceled ? "–" : availableTransferTime}
            </span>
            <span>/</span>
            <span>{transfer.transfer_time}</span>
          </div>
        </>
      )}
    </div>
  );
}
