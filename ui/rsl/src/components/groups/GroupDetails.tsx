import { UsersIcon } from "@heroicons/react/outline";
import { useAtom } from "jotai";
import React from "react";

import {
  PaxMonCompactJourneyLeg,
  PaxMonGroupRoute,
  PaxMonRerouteLogEntry,
  PaxMonRerouteReason,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonGetGroupsRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatPercent } from "@/data/numberFormat";

import { formatDateTime } from "@/util/dateFormat";

import TripServiceInfoView from "@/components/TripServiceInfoView";
import Delay from "@/components/util/Delay";

type GroupDetailsProps = {
  groupId: number;
};

function GroupDetails({ groupId }: GroupDetailsProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data, isLoading, error } = usePaxMonGetGroupsRequest({
    universe,
    ids: [groupId],
    sources: [],
    include_reroute_log: true,
  });

  if (!data) {
    if (isLoading) {
      return <div>Gruppeninformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Gruppeninformationen:{" "}
          {error instanceof Error ? error.message : `${error}`}
        </div>
      );
    }
  }
  if (data.groups.length === 0) {
    return <div>Gruppe {groupId} nicht gefunden.</div>;
  }

  const group = data.groups[0];

  if (group.routes.length === 0) {
    return <div>Gruppe {groupId} ist ungültig (keine Routen).</div>;
  }

  return (
    <div>
      <div className="flex gap-10 text-xl">
        <div>
          Reisendengruppe <span>{group.source.primary_ref}</span>
          <span className="text-db-cool-gray-400">
            .{group.source.secondary_ref}
          </span>
        </div>
        <div>Interne ID: {group.id}</div>
        <div className="flex items-center gap-x-1">
          <UsersIcon
            className="w-4 h-4 text-db-cool-gray-500"
            aria-hidden="true"
          />
          {group.passenger_count}
          <span className="sr-only">Reisende</span>
        </div>
      </div>
      <div className="">
        Planmäßige Ankunftszeit:{" "}
        {formatDateTime(group.routes[0].planned_arrival_time)}
      </div>
      <div className="mt-4">
        <div className="text-lg">
          {group.routes.length === 1
            ? "1 Route"
            : `${group.routes.length} Routen`}
        </div>
        <div className="pl-4 space-y-2">
          {group.routes.map((route) => (
            <GroupRoute route={route} key={route.index} />
          ))}
        </div>
      </div>
      <div className="mt-6">
        <div className="text-lg">Änderungsprotokoll</div>
        <div className="pl-4 space-y-2">
          {group.reroute_log.map((log, idx) => (
            <RerouteLogEntry log={log} key={idx} />
          ))}
        </div>
      </div>
    </div>
  );
}

type GroupRouteProps = {
  route: PaxMonGroupRoute;
};

function GroupRoute({ route }: GroupRouteProps): JSX.Element {
  return (
    <div>
      <div className="flex gap-4">
        <div>Route #{route.index}</div>
        <div>{formatPercent(route.probability)} Wahrscheinlichkeit</div>
        {route.planned && <div>(planmäßige Route)</div>}
      </div>
      <div className="flex gap-4">
        <div>
          Erwartete Zielverspätung: <Delay minutes={route.estimated_delay} />
        </div>
      </div>
      <div className="pl-4">
        {route.journey.legs.map((leg, idx) => (
          <JourneyLeg leg={leg} key={idx} />
        ))}
      </div>
    </div>
  );
}

type JourneyLegProps = {
  leg: PaxMonCompactJourneyLeg;
};

function JourneyLeg({ leg }: JourneyLegProps): JSX.Element {
  return (
    <div className="flex flex-wrap gap-2">
      <TripServiceInfoView tsi={leg.trip} format={"Short"} />
      <div className="space-x-1">
        <span>{formatDateTime(leg.enter_time)}</span>
        <span>{leg.enter_station.name}</span>
      </div>
      <div>&rarr;</div>
      <div className="space-x-1">
        <span>{formatDateTime(leg.exit_time)}</span>
        <span>{leg.exit_station.name}</span>
      </div>
    </div>
  );
}

function rerouteReasonText(reason: PaxMonRerouteReason): string {
  switch (reason) {
    case "Manual":
      return "Manuelle Umleitung";
    case "BrokenTransfer":
      return "Gebrochener Umstieg";
    case "MajorDelayExpected":
      return "Hohe erwartete Zielverspätung";
    case "RevertForecast":
      return "Rücknahme einer Vorhersage";
    case "Simulation":
      return "Was-wäre-wenn-Simulation";
  }
}

type RerouteLogEntryProps = {
  log: PaxMonRerouteLogEntry;
};

function RerouteLogEntry({ log }: RerouteLogEntryProps): JSX.Element {
  const broken_transfer =
    log.broken_transfer.length === 1 ? log.broken_transfer[0] : undefined;
  return (
    <div>
      <div>
        {formatDateTime(log.system_time)}: {rerouteReasonText(log.reason)}
      </div>
      <div>
        Umleitung von Route #{log.old_route.index} (
        {formatPercent(log.old_route.previous_probability)}) auf:
      </div>
      <div className="pl-4">
        {log.new_routes.map((route) => (
          <div key={route.index}>
            Route #{route.index}: {formatPercent(route.previous_probability)}{" "}
            &rarr; {formatPercent(route.new_probability)}
          </div>
        ))}
      </div>
      {broken_transfer && (
        <div>
          <div className="space-x-1">
            <span>
              Gebrochener Umstieg: Fahrtabschnitt{" "}
              {broken_transfer.leg_index + 1}
            </span>
            <span>
              ({broken_transfer.direction === "Enter" ? "Einstieg" : "Ausstieg"}
              )
            </span>
            <span>
              Benötigte Umstiegszeit: {broken_transfer.required_transfer_time}m
            </span>
          </div>
          <div className="ml-4">
            Ankunft:{" "}
            {broken_transfer.current_arrival_time !== 0
              ? formatDateTime(broken_transfer.current_arrival_time)
              : "—"}
            {broken_transfer.arrival_canceled && " (ausgefallen)"}
          </div>
          <div className="ml-4">
            {" "}
            Abfahrt:{" "}
            {broken_transfer.current_departure_time !== 0
              ? formatDateTime(broken_transfer.current_departure_time)
              : "—"}
            {broken_transfer.departure_canceled && " (ausgefallen)"}
          </div>
        </div>
      )}
    </div>
  );
}

export default GroupDetails;
