import { ClockIcon, TicketIcon } from "@heroicons/react/24/outline";
import { AlertTriangle, ExternalLink } from "lucide-react";
import React, { ReactElement } from "react";

import {
  PaxMonCompactJourney,
  PaxMonCompactJourneyLeg,
  PaxMonGroup,
  PaxMonGroupRoute,
  PaxMonTransferInfo,
} from "@/api/protocol/motis/paxmon.ts";

import { formatShortDuration } from "@/data/durationFormat.ts";
import { formatPercent } from "@/data/numberFormat.ts";

import { getBahnSucheUrl } from "@/util/bahnDe.ts";
import { formatDateTime } from "@/util/dateFormat.ts";

import TripServiceInfoView from "@/components/TripServiceInfoView.tsx";
import { Button } from "@/components/ui/button.tsx";
import Delay from "@/components/util/Delay.tsx";

import { cn } from "@/lib/utils.ts";

interface GroupRoutesProps {
  group: PaxMonGroup;
}

export function GroupRoutes({ group }: GroupRoutesProps) {
  const plannedJourney = group.routes[0].journey;

  return (
    <div className="inline-flex flex-col">
      <div className="mb-2 inline-flex items-center justify-between">
        <div className=" text-lg">
          {group.routes.length === 1
            ? "1 Route"
            : `${group.routes.length} Routen`}
        </div>
        <Button variant="outline" className="gap-2" asChild>
          <a
            href={getBahnSucheUrl(
              plannedJourney.legs[0].enter_station,
              plannedJourney.legs[plannedJourney.legs.length - 1].exit_station,
              plannedJourney.legs[0].enter_time,
            )}
            target="_blank"
            referrerPolicy="no-referrer"
            rel="noreferrer"
          >
            <ExternalLink className="h-4 w-4" aria-hidden="true" />
            Aktuelle Alternativensuche auf bahn.de
          </a>
        </Button>
      </div>
      <div className="inline-flex flex-col gap-y-3">
        {group.routes.map((route) => (
          <GroupRoute route={route} key={route.index} />
        ))}
      </div>
    </div>
  );
}

const MATCH_INEXACT_TIME = 1;
const MATCH_JOURNEY_SUBSET = 2;
const MATCH_REROUTED = 4;

interface GroupRouteProps {
  route: PaxMonGroupRoute;
}

function GroupRoute({ route }: GroupRouteProps): ReactElement {
  const showWarning = route.planned && route.source_flags != 0;
  const warnings: string[] = [];
  if ((route.source_flags & MATCH_INEXACT_TIME) == MATCH_INEXACT_TIME) {
    warnings.push(
      "Zeitstempel der Original-Reisekette weichen vom Fahrplan ab und wurden an den geladenen Fahrplan angepasst.",
    );
  }
  if ((route.source_flags & MATCH_JOURNEY_SUBSET) == MATCH_JOURNEY_SUBSET) {
    warnings.push(
      "Teil-Reisekette, da die komplette Original-Reisekette nicht dem Fahrplan zugeordnet werden konnte.",
    );
  }
  if ((route.source_flags & MATCH_REROUTED) == MATCH_REROUTED) {
    warnings.push(
      "Reisekette wurde neu berechnet, da die Original-Reisekette nicht dem Fahrplan zugeordnet werden konnte.",
    );
  }

  return (
    <div className="flex flex-col rounded bg-db-cool-gray-200 drop-shadow-md">
      <div
        className={cn(
          "grid grid-cols-3 gap-1 rounded-t border-b-4 bg-db-cool-gray-200 p-2",
          route.broken ? "border-red-300" : "border-green-300",
        )}
      >
        <div className="flex items-center gap-4 text-lg">
          #{route.index}
          {route.planned && (
            <TicketIcon className="h-5 w-5" title="Planmäßge Route" />
          )}
          {showWarning && (
            <div title={warnings.join("\n")}>
              <AlertTriangle className="h-5 w-5 text-orange-600" />
            </div>
          )}
        </div>
        <div className="text-center text-lg">
          {formatPercent(route.probability)}
        </div>
        <div
          className="flex items-center justify-end gap-1"
          title="Erwartete Zielverspätung"
        >
          <ClockIcon className="h-5 w-5" />
          <Delay minutes={route.estimated_delay} forceSign={true} />
        </div>
      </div>
      <div
        className={cn(
          "rounded-b p-2",
          route.broken
            ? "bg-red-50"
            : route.probability > 0
              ? "bg-green-50"
              : "bg-amber-50",
        )}
      >
        <table>
          <thead>
            <tr className="font-semibold">
              <td className="sr-only pr-2">Abschnitt</td>
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
            <FinalFootpath journey={route.journey} />
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface JourneyLegProps {
  leg: PaxMonCompactJourneyLeg;
  index: number;
}

function JourneyLeg({ leg, index }: JourneyLegProps): JSX.Element {
  return (
    <tr>
      <td className="pr-2">{index + 1}.</td>
      <td className="pr-2">
        <TripServiceInfoView tsi={leg.trip} format={"ShortAll"} link={true} />
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

interface FinalFootpathProps {
  journey: PaxMonCompactJourney;
}

function FinalFootpath({ journey }: FinalFootpathProps) {
  if (journey.final_footpath.length === 1 && journey.legs.length > 0) {
    const fp = journey.final_footpath[0];
    const lastLeg = journey.legs[journey.legs.length - 1];
    return (
      <tr>
        <td className="pr-2">{journey.legs.length + 1}.</td>
        <td className="pr-2">Fußweg</td>
        <td className="pr-2">{formatShortDuration(fp.duration)}</td>
        <td className="pr-2">{formatDateTime(lastLeg.exit_time)}</td>
        <td className="pr-2" title={fp.from_station.id}>
          {fp.from_station.name}
        </td>
        <td className="pr-2">
          {formatDateTime(lastLeg.exit_time + 60 * fp.duration)}
        </td>
        <td className="" title={fp.to_station.id}>
          {fp.to_station.name}
        </td>
      </tr>
    );
  } else {
    return null;
  }
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
