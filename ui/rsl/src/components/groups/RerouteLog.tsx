import {
  ArrowPathIcon,
  ArrowUturnUpIcon,
  CheckCircleIcon,
  ClockIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  WrenchIcon,
  XCircleIcon,
} from "@heroicons/react/24/outline";
import React, { ReactElement, ReactNode } from "react";

import {
  PaxMonAtStation,
  PaxMonGroup,
  PaxMonInTrip,
  PaxMonRerouteLogEntry,
  PaxMonRerouteLogRoute,
  PaxMonRerouteReason,
} from "@/api/protocol/motis/paxmon.ts";

import { formatShortDuration } from "@/data/durationFormat.ts";
import { formatPercent } from "@/data/numberFormat.ts";

import { formatDateTime, formatTime } from "@/util/dateFormat.ts";

import { cn } from "@/lib/utils.ts";

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

interface RerouteLogEntryProps {
  log: PaxMonRerouteLogEntry;
  logIndex: number;
  group: PaxMonGroup;
}

export function RerouteLogEntry({
  log,
  logIndex,
}: RerouteLogEntryProps): ReactNode {
  const broken_transfer =
    log.broken_transfer.length === 1 ? log.broken_transfer[0] : undefined;
  const show_reroutes = log.new_routes.length > 0;
  const { icon, bgColor } = getRerouteReasonIcon(log.reason);

  return (
    <div className="relative">
      <div
        className={cn(
          "absolute inline-flex h-8 w-8 items-center justify-center rounded-full text-white",
          bgColor,
        )}
      >
        {icon}
      </div>
      <div className="ml-12 w-auto pt-1">
        <div className="flex justify-between">
          <span className="font-semibold">
            V{logIndex + 1}: {rerouteReasonText(log.reason)}
          </span>
          <span>Update {log.update_number}</span>
          <span className="text-db-cool-gray-500">
            {formatDateTime(log.system_time)}
          </span>
        </div>
        <RerouteLogEntryLocalization logRoute={log.old_route} />
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

interface RerouteLogEntryLocalizationProps {
  logRoute: PaxMonRerouteLogRoute;
}

function RerouteLogEntryLocalization({
  logRoute,
}: RerouteLogEntryLocalizationProps): ReactNode {
  switch (logRoute.localization_type) {
    case "PaxMonAtStation": {
      const loc = logRoute.localization as PaxMonAtStation;
      return (
        <div>
          Reisende an Station {loc.station.name} um{" "}
          {formatTime(loc.current_arrival_time)}
          {loc.first_station ? " (Reisebeginn)" : ""}
        </div>
      );
    }
    case "PaxMonInTrip": {
      const loc = logRoute.localization as PaxMonInTrip;
      return (
        <div>
          Reisende in Zug {loc.trip.train_nr}, nächster Halt:{" "}
          {loc.next_station.name} um {formatTime(loc.current_arrival_time)}
        </div>
      );
    }
  }
}

interface RerouteReasonIcon {
  icon: ReactElement;
  bgColor: string;
}

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
        bgColor: "bg-teal-500",
      };
    case "Simulation":
      return {
        icon: <CpuChipIcon className={style} />,
        bgColor: "bg-cyan-500",
      };
    case "UpdateForecast":
      return {
        icon: <ArrowPathIcon className={style} />,
        bgColor: "bg-violet-500",
      };
    case "DestinationUnreachable":
      return {
        icon: <XCircleIcon className={style} />,
        bgColor: "bg-fuchsia-500",
      };
    case "DestinationReachable":
      return {
        icon: <CheckCircleIcon className={style} />,
        bgColor: "bg-lime-500",
      };
  }
}

interface RerouteLogTableProps {
  group: PaxMonGroup;
}

export function RerouteLogTable({ group }: RerouteLogTableProps) {
  const probs: number[][] = [group.routes.map((r) => (r.index === 0 ? 1 : 0))];
  const diffs: number[][] = [group.routes.map(() => 0)];

  for (const le of group.reroute_log) {
    const new_probs = [...probs[probs.length - 1]];
    const diff = group.routes.map(() => 0);

    new_probs[le.old_route.index] = le.old_route.new_probability;
    diff[le.old_route.index] =
      le.old_route.new_probability - le.old_route.previous_probability;
    for (const nr of le.new_routes) {
      new_probs[nr.index] = nr.new_probability;
      diff[nr.index] = nr.new_probability - nr.previous_probability;
    }

    probs.push(new_probs);
    diffs.push(diff);
  }

  return (
    <table className="mt-2">
      <thead>
        <tr className="font-semibold">
          <td className="sr-only pr-4">V</td>
          {group.routes.map((r) => (
            <td key={r.index} className="pr-4 text-center">
              R #{r.index}
            </td>
          ))}
          <td className="pr-4 text-center">Summe</td>
        </tr>
      </thead>
      <tbody>
        {probs.map((row, rowIdx) => (
          <tr key={rowIdx}>
            <td className="pr-4 font-semibold">V{rowIdx}</td>
            {row.map((p, colIdx) => (
              <td
                key={colIdx}
                className={cn(
                  "pr-4 text-center",
                  p === 0
                    ? diffs[rowIdx][colIdx] < 0
                      ? "text-db-red-300"
                      : "text-db-cool-gray-300"
                    : diffs[rowIdx][colIdx] > 0
                      ? "text-green-600"
                      : diffs[rowIdx][colIdx] < 0
                        ? "text-yellow-500"
                        : "text-black",
                )}
                title={`Exakter Wert: ${p}, Änderung: ${formatPercent(
                  diffs[rowIdx][colIdx],
                )} (${diffs[rowIdx][colIdx]})`}
              >
                {formatPercent(p)}
              </td>
            ))}
            <td className="pr-4 text-center">
              {formatPercent(row.reduce((acc, p) => acc + p, 0))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export interface RerouteLogProps {
  group: PaxMonGroup;
}

export function RerouteLog({ group }: RerouteLogProps) {
  return (
    <div>
      <div className="mb-3 text-lg">Änderungsprotokoll</div>
      <div className="relative inline-flex flex-col">
        <div className="absolute left-4 top-0 -ml-px h-full w-0.5 bg-db-cool-gray-300"></div>
        {group.reroute_log.map((log, idx) => (
          <RerouteLogEntry key={idx} log={log} logIndex={idx} group={group} />
        ))}
      </div>
      <div className="mb-3 mt-10 text-lg">
        Routenwahrscheinlichkeiten im Zeitverlauf
      </div>
      <RerouteLogTable group={group} />
    </div>
  );
}
