import {
  ArrowPathIcon,
  ArrowUturnUpIcon,
  CheckCircleIcon,
  ClockIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  TicketIcon,
  UsersIcon,
  WrenchIcon,
  XCircleIcon,
} from "@heroicons/react/24/outline";
import { useAtom } from "jotai";
import { useUpdateAtom } from "jotai/utils";
import React, { useEffect } from "react";
import { Link, useParams } from "react-router-dom";

import {
  PaxMonAtStation,
  PaxMonCompactJourneyLeg,
  PaxMonGroup,
  PaxMonGroupRoute,
  PaxMonInTrip,
  PaxMonRerouteLogEntry,
  PaxMonRerouteReason,
  PaxMonTransferInfo,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonGetGroupsRequest } from "@/api/paxmon";

import { formatShortDuration } from "@/data/durationFormat";
import { universeAtom } from "@/data/multiverse";
import { formatPercent } from "@/data/numberFormat";
import { mostRecentlySelectedGroupAtom } from "@/data/selectedGroup";

import classNames from "@/util/classNames";
import { formatDateTime, formatTime } from "@/util/dateFormat";

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

  const setMostRecentlySelectedGroup = useUpdateAtom(
    mostRecentlySelectedGroupAtom
  );
  useEffect(() => {
    setMostRecentlySelectedGroup(groupId);
  }, [groupId, setMostRecentlySelectedGroup]);

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
      <div className="mt-4 flex flex-wrap gap-8">
        <div>
          <div className="text-lg">
            {group.routes.length === 1
              ? "1 Route"
              : `${group.routes.length} Routen`}
          </div>
          <div className="inline-flex flex-col gap-y-3">
            {group.routes.map((route) => (
              <GroupRoute route={route} key={route.index} />
            ))}
          </div>
        </div>
        <div>
          <div className="text-lg mb-3">Änderungsprotokoll</div>
          <div className="relative inline-flex flex-col">
            <div className="absolute top-0 left-4 h-full -ml-px bg-db-cool-gray-300 w-0.5"></div>
            {group.reroute_log.map((log, idx) => (
              <RerouteLogEntry
                key={idx}
                log={log}
                logIndex={idx}
                group={group}
              />
            ))}
          </div>
          <RerouteLogTable group={group} />
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
    <div className="flex flex-col bg-db-cool-gray-200 rounded drop-shadow-md">
      <div
        className={classNames(
          "grid grid-cols-3 gap-1 rounded-t p-2 bg-db-cool-gray-200 border-b-4",
          route.broken ? "border-red-300" : "border-green-300"
        )}
      >
        <div className="text-lg flex items-center gap-4">
          #{route.index}
          {route.planned && (
            <TicketIcon className="w-5 h-5" title="Planmäßge Route" />
          )}
        </div>
        <div className="text-lg text-center">
          {formatPercent(route.probability)}
        </div>
        <div
          className="flex justify-end items-center gap-1"
          title="Erwartete Zielverspätung"
        >
          <ClockIcon className="w-5 h-5" />
          <Delay minutes={route.estimated_delay} forceSign={true} />
        </div>
      </div>
      <div
        className={classNames(
          "rounded-b p-2",
          route.broken
            ? "bg-red-50"
            : route.probability > 0
            ? "bg-green-50"
            : "bg-amber-50"
        )}
      >
        <table>
          <thead>
            <tr className="font-semibold">
              <td className="pr-2 sr-only">Abschnitt</td>
              <td className="pr-2">Zug</td>
              <td className="pr-2" title="Benötigte Umstiegszeit">
                Umstieg
              </td>
              <td className="pr-2" colSpan={2}>
                Planmäßige Abfahrt
              </td>
              <td className="pr-2" colSpan={2}>
                Planmäßige Ankunft
              </td>
            </tr>
          </thead>
          <tbody>
            {route.journey.legs.map((leg, idx) => (
              <JourneyLeg key={idx} leg={leg} index={idx} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type JourneyLegProps = {
  leg: PaxMonCompactJourneyLeg;
  index: number;
};

function JourneyLeg({ leg, index }: JourneyLegProps): JSX.Element {
  return (
    <tr>
      <td className="pr-2">{index + 1}.</td>
      <td className="pr-2">
        <Link
          to={`/trips/${encodeURIComponent(JSON.stringify(leg.trip.trip))}`}
        >
          <TripServiceInfoView tsi={leg.trip} format={"ShortAll"} />
        </Link>
      </td>
      <td className="pr-2" title={transferTypeText(leg.enter_transfer)}>
        {requiresTransfer(leg.enter_transfer)
          ? formatShortDuration(leg.enter_transfer.duration)
          : "—"}
      </td>
      <td className="pr-2">{formatDateTime(leg.enter_time)}</td>
      <td className="pr-2" title={leg.enter_station.id}>
        {leg.enter_station.name}
      </td>
      <td className="pr-2">{formatDateTime(leg.exit_time)}</td>
      <td className="" title={leg.exit_station.id}>
        {leg.exit_station.name}
      </td>
    </tr>
  );
}

function requiresTransfer(ti: PaxMonTransferInfo) {
  return ti.type === "SAME_STATION" || ti.type === "FOOTPATH";
}

function transferTypeText(ti: PaxMonTransferInfo) {
  switch (ti.type) {
    case "NONE":
      return "Kein Umstieg";
    case "SAME_STATION":
      return "Umstieg an einer Station";
    case "FOOTPATH":
      return "Umstieg mit Fußweg zwischen zwei Stationen";
    case "MERGE":
      return "Vereinigung";
    case "THROUGH":
      return "Durchbindung";
  }
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
      return "Simulation";
    case "UpdateForecast":
      return "Neuberechnung der Vorhersage";
    case "DestinationUnreachable":
      return "Ziel nicht mehr erreichbar";
    case "DestinationReachable":
      return "Ziel wieder erreichbar";
  }
}

type RerouteLogEntryProps = {
  log: PaxMonRerouteLogEntry;
  logIndex: number;
  group: PaxMonGroup;
};

function RerouteLogEntry({ log, logIndex }: RerouteLogEntryProps): JSX.Element {
  const broken_transfer =
    log.broken_transfer.length === 1 ? log.broken_transfer[0] : undefined;
  const show_reroutes = log.new_routes.length > 0;
  const { icon, bgColor } = getRerouteReasonIcon(log.reason);

  return (
    <div className="relative">
      <div
        className={classNames(
          "absolute w-8 h-8 rounded-full inline-flex items-center justify-center text-white",
          bgColor
        )}
      >
        {icon}
      </div>
      <div className="ml-12 w-auto pt-1">
        <div className="flex justify-between">
          <span className="font-semibold">
            V{logIndex + 1}: {rerouteReasonText(log.reason)}
          </span>
          <span className="text-db-cool-gray-500">
            {formatDateTime(log.system_time)}
          </span>
        </div>
        <RerouteLogEntryLocalization log={log} />
        {show_reroutes ? (
          <>
            <div>
              Umleitung von Route #{log.old_route.index} (
              {formatPercent(log.old_route.previous_probability)} &rarr;{" "}
              {formatPercent(log.old_route.new_probability)}) auf:
            </div>
            <div className="pl-4">
              {log.new_routes.map((route) => (
                <div key={route.index}>
                  Route #{route.index}:{" "}
                  {formatPercent(route.previous_probability)} &rarr;{" "}
                  {formatPercent(route.new_probability)}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div>
            Route #{log.old_route.index} (
            {formatPercent(log.old_route.previous_probability)})
          </div>
        )}
        {broken_transfer && (
          <div>
            <div className="space-x-1">
              <span>
                Gebrochener Umstieg: Fahrtabschnitt{" "}
                {broken_transfer.leg_index + 1}
              </span>
              <span>
                (
                {broken_transfer.direction === "Enter"
                  ? "Einstieg"
                  : "Ausstieg"}
                )
              </span>
              <span>
                Benötigte Umstiegszeit:{" "}
                {formatShortDuration(broken_transfer.required_transfer_time)}
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
    </div>
  );
}

type RerouteLogEntryLocalizationProps = {
  log: PaxMonRerouteLogEntry;
};

function RerouteLogEntryLocalization({
  log,
}: RerouteLogEntryLocalizationProps): JSX.Element {
  switch (log.localization_type) {
    case "PaxMonAtStation": {
      const loc = log.localization as PaxMonAtStation;
      return (
        <div>
          Reisende an Station {loc.station.name} um{" "}
          {formatTime(loc.current_arrival_time)}
          {loc.first_station ? " (Reisebeginn)" : ""}
        </div>
      );
    }
    case "PaxMonInTrip": {
      const loc = log.localization as PaxMonInTrip;
      return (
        <div>
          Reisende in Zug {loc.trip.train_nr}, nächster Halt:{" "}
          {loc.next_station.name} um {formatTime(loc.current_arrival_time)}
        </div>
      );
    }
  }
}

type RerouteReasonIcon = {
  icon: JSX.Element;
  bgColor: string;
};

function getRerouteReasonIcon(reason: PaxMonRerouteReason): RerouteReasonIcon {
  const style = "w-6 h-6";
  switch (reason) {
    case "Manual":
      return {
        icon: <WrenchIcon className={style} />,
        bgColor: "bg-blue-500",
      };
    case "BrokenTransfer":
      return {
        icon: <ExclamationTriangleIcon className={style} />,
        bgColor: "bg-red-500",
      };
    case "MajorDelayExpected":
      return {
        icon: <ClockIcon className={style} />,
        bgColor: "bg-amber-500",
      };
    case "RevertForecast":
      return {
        icon: <ArrowUturnUpIcon className={style} />,
        bgColor: "bg-green-500",
      };
    case "Simulation":
      return {
        icon: <CpuChipIcon className={style} />,
        bgColor: "bg-teal-500",
      };
    case "UpdateForecast":
      return {
        icon: <ArrowPathIcon className={style} />,
        bgColor: "bg-teal-500",
      };
    case "DestinationUnreachable":
      return {
        icon: <XCircleIcon className={style} />,
        bgColor: "bg-fuchsia-500",
      };
    case "DestinationReachable":
      return {
        icon: <CheckCircleIcon className={style} />,
        bgColor: "bg-green-500",
      };
  }
}

type RerouteLogTableProps = {
  group: PaxMonGroup;
};

function RerouteLogTable({ group }: RerouteLogTableProps): JSX.Element {
  const probs = [group.routes.map((r) => r.probability)];
  const diffs: Array<Array<number>> = [];

  for (let i = group.reroute_log.length - 1; i >= 0; --i) {
    const le = group.reroute_log[i];
    const new_probs = [...probs[0]];
    const diff = group.routes.map(() => 0);
    new_probs[le.old_route.index] = le.old_route.previous_probability;
    diff[le.old_route.index] = -le.old_route.previous_probability;
    for (const nr of le.new_routes) {
      new_probs[nr.index] = nr.previous_probability;
      diff[nr.index] = nr.new_probability - nr.previous_probability;
    }
    probs.unshift(new_probs);
    diffs.unshift(diff);
  }
  diffs.unshift(group.routes.map(() => 0));

  return (
    <table className="mt-2">
      <thead>
        <tr className="font-semibold">
          <td className="pr-4 sr-only">V</td>
          {group.routes.map((r) => (
            <td key={r.index} className="pr-4 text-center">
              R #{r.index}
            </td>
          ))}
        </tr>
      </thead>
      <tbody>
        {probs.map((row, rowIdx) => (
          <tr key={rowIdx}>
            <td className="pr-4 font-semibold">V{rowIdx}</td>
            {row.map((p, colIdx) => (
              <td
                key={colIdx}
                className={classNames(
                  "pr-4 text-center",
                  p === 0
                    ? diffs[rowIdx][colIdx] < 0
                      ? "text-db-red-300"
                      : "text-db-cool-gray-300"
                    : diffs[rowIdx][colIdx] > 0
                    ? "text-green-600"
                    : diffs[rowIdx][colIdx] < 0
                    ? "text-yellow-500"
                    : "text-black"
                )}
                title={`Exakter Wert: ${p}, Änderung: ${formatPercent(
                  diffs[rowIdx][colIdx]
                )} (${diffs[rowIdx][colIdx]})`}
              >
                {formatPercent(p)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function GroupDetailsFromRoute(): JSX.Element {
  const params = useParams();
  const groupId = Number.parseInt(params.groupId ?? "");
  if (!Number.isNaN(groupId)) {
    return <GroupDetails groupId={groupId} key={groupId} />;
  } else {
    return <></>;
  }
}

export default GroupDetails;
