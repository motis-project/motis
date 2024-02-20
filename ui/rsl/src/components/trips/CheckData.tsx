import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowBigLeft,
  ArrowBigRight,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Asterisk,
  CheckCircle2,
  HelpCircle,
  Info,
  XCircle,
} from "lucide-react";
import React, { ReactNode } from "react";

import { Station, TripId } from "@/api/protocol/motis.ts";
import {
  PaxMonCheckDataByOrderRequest,
  PaxMonCheckDataRequest,
  PaxMonCheckDataResponse,
  PaxMonCheckDirection,
  PaxMonCheckEntry,
  PaxMonCheckLegStatus,
  PaxMonCheckSectionData,
  PaxMonCheckType,
  PaxMonTripLoadInfo,
} from "@/api/protocol/motis/paxmon.ts";

import {
  queryKeys,
  sendPaxMonCheckDataByOrderRequest,
  sendPaxMonCheckDataRequest,
  sendPaxMonGetTripLoadInfosRequest,
} from "@/api/paxmon.ts";

import { formatNumber } from "@/data/numberFormat.ts";

import { formatDateTime, formatTime } from "@/util/dateFormat.ts";

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card.tsx";

import { cn } from "@/lib/utils.ts";

export interface CheckDataProps {
  tripId: TripId;
}

function optionalTime(ts: number) {
  return ts !== 0 ? formatDateTime(ts) : "—";
}

function optionalStation(station: Station | null | undefined) {
  return station ? station.name : "—";
}

function shortCheckType(t: PaxMonCheckType) {
  switch (t) {
    case "NOT_CHECKED":
      return "N. K.";
    case "TICKED_CHECKED":
      return "K";
    case "CHECKIN":
      return "CI";
    case "BOTH":
      return "K + CI";
  }
}

function longCheckType(t: PaxMonCheckType) {
  switch (t) {
    case "NOT_CHECKED":
      return "Nicht kontrolliert";
    case "TICKED_CHECKED":
      return "Kontrolle";
    case "CHECKIN":
      return "Check-in";
    case "BOTH":
      return "Kontrolle + Check-in";
  }
}

function shortLegStatus(s: PaxMonCheckLegStatus) {
  switch (s) {
    case "NOT_CHECKED_COVERED":
      return "0: NK, A";
    case "CHECKED_PLANNED":
      return "1: K, P=I";
    case "CHECKED_DEVIATION_EXACT_MATCH":
      return "2: K, P!=I";
    case "CHECKED_DEVIATION_EQUIVALENT_MATCH":
      return "3: K, P!=I";
    case "CHECKED_DEVIATION_NO_MATCH":
      return "4: K, P!=I";
    case "NOT_CHECKED_NOT_COVERED":
      return "5: NK, NA";
  }
}

function longLegStatus(s: PaxMonCheckLegStatus) {
  switch (s) {
    case "NOT_CHECKED_COVERED":
      return "0: Nicht kontrolliert, durch anderen kontrollierten Abschnitt abgedeckt";
    case "CHECKED_PLANNED":
      return "1: Kontrolliert, Plan = Ist";
    case "CHECKED_DEVIATION_EXACT_MATCH":
      return "2: Kontrolliert, Plan <> Ist, passender Abschnitt mit identischen Bahnhöfen";
    case "CHECKED_DEVIATION_EQUIVALENT_MATCH":
      return "3: Kontrolliert, Plan <> Ist, passender Abschnitt mit äquivalenten Bahnhöfen";
    case "CHECKED_DEVIATION_NO_MATCH":
      return "4: Kontrolliert, Plan <> Ist, kein passender Abschnitt";
    case "NOT_CHECKED_NOT_COVERED":
      return "5: Nicht kontrolliert, nicht abgedeckt";
  }
}

export function CheckData({ tripId }: CheckDataProps) {
  const request: PaxMonCheckDataRequest = { trip_id: tripId };
  const {
    data: checkData,
    isPending,
    error,
  } = useQuery({
    queryKey: queryKeys.checkData(request),
    queryFn: () => sendPaxMonCheckDataRequest(request),
  });

  const { data: loadData } = useQuery({
    queryKey: queryKeys.tripLoad(0, tripId),
    queryFn: () =>
      sendPaxMonGetTripLoadInfosRequest({ universe: 0, trips: [tripId] }),
  });

  if (!checkData) {
    if (isPending) {
      return <div>Reisendenzähldaten werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Reisendenzähldaten:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
        </div>
      );
    }
  }

  return (
    <div className="py-4">
      <h2 className="mb-1 text-2xl">
        Reisendenzähldaten für {checkData.category} {checkData.train_nr}
      </h2>
      <div className="mb-4">
        Einträge: {formatNumber(checkData.matched_entry_count)} passend,{" "}
        {formatNumber(checkData.unmatched_entry_count)} unpassend
      </div>
      <CheckDataBySection
        checkData={checkData}
        loadData={loadData ? loadData.load_infos[0] : undefined}
        includeSectionDetails={true}
      />
      <details className="my-4">
        <summary className=" cursor-pointer text-xl">
          Alle Reisendenzähldaten für {checkData.category} {checkData.train_nr}
        </summary>
        <CheckEntries entries={checkData.entries} />
      </details>
    </div>
  );
}

interface CheckEntriesProps {
  entries: PaxMonCheckEntry[];
  section?: PaxMonCheckSectionData | undefined;
}

function CheckEntries({ entries, section }: CheckEntriesProps) {
  const thClass =
    "sticky top-0 bg-white py-1 border-b-2 border-db-cool-gray-300 font-semibold";

  return (
    <table className="relative border-separate text-xs">
      <thead>
        <tr className="text-center">
          {/*<td className={thClass}>Ref</td>*/}
          <th className={cn(thClass, "border-none")}></th>
          <th className={thClass}>Typ</th>
          <th className={thClass} title="Anzahl Reisende">
            #Rsd
          </th>
          <th className={thClass} title="Anzahl Kontrollen">
            #Ktr
          </th>
          <th className={thClass} title="Planmäßig?">
            Plan
          </th>
          <th className={thClass} title="Kontrolliert?">
            Ktr
          </th>
          <th className={thClass} title="Ausfall?">
            Ausf
          </th>
          <th className={thClass} title="Reiseabschnitt-Status">
            RA Status
          </th>
          <th className={thClass} colSpan={2}>
            Reiseabschnitt Start
          </th>
          <th className={thClass} colSpan={2}>
            Reiseabschnitt Ziel
          </th>
          <th className={thClass}>Check-In Start</th>
          <th className={thClass}>Check-In Ziel</th>
          <th className={thClass} title="Frühester Kontrollzeitpunkt">
            Ktr Min
          </th>
          <th className={thClass} title="Spätester Kontrollzeitpunkt">
            Ktr Max
          </th>
          <th className={thClass}>Plan Zug Start</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((entry) => (
          <CheckEntryRow key={entry.ref} entry={entry} section={section} />
        ))}
      </tbody>
    </table>
  );
}

interface CheckEntryRowProps {
  entry: PaxMonCheckEntry;
  section?: PaxMonCheckSectionData | undefined;
}

function CheckEntryRow({ entry, section }: CheckEntryRowProps) {
  const hasCheckInfo =
    section && entry.check_min_time != 0 && entry.check_max_time != 0;
  const checkMinInSection =
    hasCheckInfo &&
    entry.check_min_time >= section.departure_current_time &&
    entry.check_min_time <= section.arrival_current_time;
  const checkMaxInSection =
    hasCheckInfo &&
    entry.check_max_time >= section.departure_current_time &&
    entry.check_max_time <= section.arrival_current_time;
  const hasMultipleChecks = entry.check_min_time != entry.check_max_time;
  const isCheck =
    entry.check_type == "TICKED_CHECKED" || entry.check_type == "BOTH";

  return (
    <tr
      className={cn(
        entry.leg_status == "NOT_CHECKED_COVERED"
          ? "text-gray-500"
          : "text-black",
      )}
    >
      <td className="pr-1 text-center">
        <HoverCard openDelay={200}>
          <HoverCardTrigger asChild>
            <Info className="h-4 w-4 text-gray-500 transition hover:text-black" />
          </HoverCardTrigger>
          <HoverCardContent className="max-h-[32rem] w-max overflow-y-auto">
            <OrderCheckDataCard entryRef={entry.ref} orderId={entry.order_id} />
          </HoverCardContent>
        </HoverCard>
      </td>
      <td className="pr-2 text-center">{shortCheckType(entry.check_type)}</td>
      <td className="pr-2 text-center">{entry.passengers}</td>
      <td className="pr-2 text-center">{entry.check_count}</td>
      <td className="pr-2 text-center">{entry.planned_train ? "+" : "-"}</td>
      <td className="pr-2 text-center">{entry.checked_in_train ? "+" : "-"}</td>
      <td className="pr-2 text-center">{entry.canceled ? "+" : "-"}</td>
      <td className="pr-2">{shortLegStatus(entry.leg_status)}</td>
      <td className="pr-2">{optionalStation(entry.leg_start_station)}</td>
      <td className="pr-2">{optionalTime(entry.leg_start_time)}</td>
      <td className="pr-2">{optionalStation(entry.leg_destination_station)}</td>
      <td className="pr-2">{optionalTime(entry.leg_destination_time)}</td>
      <td
        className={cn(
          "pr-2",
          entry.checkin_start_station &&
            entry.checkin_start_station?.id != entry.leg_start_station?.id &&
            "text-red-700",
        )}
      >
        {optionalStation(entry.checkin_start_station)}
      </td>
      <td
        className={cn(
          "pr-2",
          entry.checkin_destination_station &&
            entry.checkin_destination_station?.id !=
              entry.leg_destination_station?.id &&
            "text-red-700",
        )}
      >
        {optionalStation(entry.checkin_destination_station)}
      </td>
      <td
        className={cn("pr-2", checkMinInSection && isCheck && "text-green-600")}
      >
        <div className="flex items-center gap-1">
          {optionalTime(entry.check_min_time)}
          {hasCheckInfo &&
            entry.check_min_time < section?.departure_current_time && (
              <ArrowUp className="h-4 w-4 text-gray-800" />
            )}
          {hasCheckInfo &&
            entry.check_min_time > section?.arrival_current_time && (
              <ArrowDown className="h-4 w-4 text-gray-800" />
            )}
        </div>
      </td>
      <td
        className={cn(
          "pr-2",
          !hasMultipleChecks && "text-gray-300",
          hasMultipleChecks && checkMaxInSection && isCheck && "text-green-600",
        )}
      >
        <div className="flex items-center gap-1">
          {optionalTime(entry.check_max_time)}
          {hasCheckInfo &&
            entry.check_max_time < section?.departure_current_time && (
              <ArrowUp
                className={cn("h-4 w-4", hasMultipleChecks && "text-gray-800")}
              />
            )}
          {hasCheckInfo &&
            entry.check_max_time > section?.arrival_current_time && (
              <ArrowDown
                className={cn("h-4 w-4", hasMultipleChecks && "text-gray-800")}
              />
            )}
        </div>
      </td>
      <td className="pr-2">{optionalTime(entry.schedule_train_start_time)}</td>
    </tr>
  );
}

function sqr(x: number) {
  return x * x;
}

interface CheckDataBySectionProps {
  checkData: PaxMonCheckDataResponse;
  loadData: PaxMonTripLoadInfo | undefined;
  includeSectionDetails?: boolean | undefined;
}

function CheckDataBySection({
  checkData,
  loadData,
  includeSectionDetails,
}: CheckDataBySectionProps) {
  if (!loadData) {
    return <div>Auslastungsprognose wird geladen...</div>;
  }
  if (checkData.sections.length != loadData.edges.length) {
    return (
      <div>
        Daten der Auslastungsprognose und Zähldaten passen nicht zusammen
        (unterschiedliche Anzahl Fahrtabschnitte).
      </div>
    );
  }

  const sectionSummary: ReactNode[] = [];
  const sectionDetails: ReactNode[] = [];

  const formatFactor = (factor: number) =>
    formatNumber(factor, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }) + "x";

  const formatDiff = (diff: number) =>
    formatNumber(diff, { signDisplay: "always" });

  let maePlan = 0;
  let msePlan = 0;
  let maeQ50 = 0;
  let mseQ50 = 0;
  let sectionsWithChecks = 0;

  for (let i = 0; i < checkData.sections.length; i++) {
    const cd = checkData.sections[i];
    const ld = loadData.edges[i];

    const departureDelayed =
      cd.departure_current_time > cd.departure_schedule_time;
    const arrivalDelayed = cd.arrival_current_time > cd.arrival_schedule_time;

    const diffPlan = ld.expected_passengers - cd.checked_pax_count;
    const diffQ50 = ld.dist.q50 - cd.checked_pax_count;

    if (cd.check_count != 0) {
      ++sectionsWithChecks;
      maePlan += Math.abs(diffPlan);
      msePlan += sqr(diffPlan);
      maeQ50 += Math.abs(diffQ50);
      mseQ50 += sqr(diffQ50);
    }

    sectionSummary.push(
      <tr key={i}>
        <td className="max-w-40 truncate pr-1">{cd.from.name}</td>
        <td className="flex gap-1 px-1">
          <span>{formatDateTime(cd.departure_schedule_time)}</span>
          <span
            className={cn(departureDelayed ? "text-red-600" : "text-green-600")}
          >
            {formatTime(cd.departure_current_time)}
          </span>
        </td>
        <td className="max-w-40 truncate">{cd.to.name}</td>
        <td className="flex gap-1 px-1">
          <span>{formatDateTime(cd.arrival_schedule_time)}</span>
          <span
            className={cn(arrivalDelayed ? "text-red-600" : "text-green-600")}
          >
            {formatTime(cd.arrival_current_time)}
          </span>
        </td>
        <td className="px-1">
          {formatNumber(
            (cd.arrival_current_time - cd.departure_current_time) / 60,
            { style: "unit", unit: "minute" },
          )}
        </td>
        <td className="bg-red-100 px-1 text-center">{cd.total_pax_count}</td>
        <td className="bg-red-100 px-1 text-center font-semibold">
          {cd.min_pax_count}
        </td>
        <td className="bg-red-100 px-1 text-center font-semibold">
          {cd.avg_pax_count}
        </td>
        <td
          className="bg-red-100 px-1 text-center font-semibold"
          title={`${cd.checked_pax_count} kontrolliert\n+ ${cd.unchecked_uncovered_pax_count} nicht kontrolliert, aber nicht anderweitig abgedeckt\n+ ${cd.possible_additional_pax_count} ohne Abschnittsinformationen`}
        >
          {cd.max_pax_count}
        </td>
        <td
          className={cn(
            "bg-red-100 px-1 text-center",
            cd.check_count == 0 && "text-red-600",
          )}
        >
          {cd.check_count}
        </td>
        <td className="bg-blue-100 px-1 text-center">
          {ld.expected_passengers}
        </td>
        <td className="bg-blue-100 px-1 text-center">{ld.dist.q5}</td>
        <td className="bg-blue-100 px-1 text-center font-semibold">
          {ld.dist.q50}
        </td>
        <td className="bg-blue-100 px-1 text-center">{ld.dist.q95}</td>
        <td className="bg-yellow-100 px-1 text-center">
          {formatDiff(ld.expected_passengers - cd.avg_pax_count)}
        </td>
        <td className="bg-yellow-100 px-1 text-center">
          {formatFactor(ld.expected_passengers / cd.avg_pax_count)}
        </td>
        <td className="bg-green-100 px-1 text-center">
          {formatDiff(ld.dist.q5 - cd.min_pax_count)}
        </td>
        <td className="bg-green-100 px-1 text-center">
          {formatFactor(ld.dist.q5 / cd.min_pax_count)}
        </td>
        <td className="bg-green-200 px-1 text-center">
          {formatDiff(ld.dist.q50 - cd.avg_pax_count)}
        </td>
        <td className="bg-green-200 px-1 text-center">
          {formatFactor(ld.dist.q50 / cd.avg_pax_count)}
        </td>
        <td className="bg-green-300 px-1 text-center">
          {formatDiff(ld.dist.q95 - cd.max_pax_count)}
        </td>
        <td className="bg-green-300 px-1 text-center">
          {formatFactor(ld.dist.q95 / cd.max_pax_count)}
        </td>

        {cd.check_count == 0 ? (
          <td title="Auf diesem Fahrtabschnitt wurden keine Reisenden kontrolliert">
            <AlertTriangle className="ml-1 h-4 w-4 text-red-800" />
          </td>
        ) : (
          <td></td>
        )}
      </tr>,
    );

    if (includeSectionDetails) {
      const filteredEntries = checkData.entries.filter((entry) =>
        cd.entry_refs.includes(entry.ref),
      );
      sectionDetails.push(
        <details key={i} className="mb-4">
          <summary className=" cursor-pointer text-xl">
            {ld.from.name} → {ld.to.name}
          </summary>
          <div className="my-2 text-sm">
            <span className="mr-5">
              {formatDateTime(ld.departure_current_time)} →{" "}
              {formatDateTime(ld.arrival_current_time)}
            </span>
            <span>
              (planmäßig {formatDateTime(ld.departure_schedule_time)} -{" "}
              {formatDateTime(ld.arrival_schedule_time)})
            </span>
          </div>
          <CheckEntries entries={filteredEntries} section={cd} />
        </details>,
      );
    }
  }

  if (sectionsWithChecks != 0) {
    maePlan /= sectionsWithChecks;
    msePlan /= sectionsWithChecks;
    maeQ50 /= sectionsWithChecks;
    mseQ50 /= sectionsWithChecks;
  }

  const thClass = "py-1 px-1 border-b-2 border-db-cool-gray-300 font-semibold";
  const thTopClass = "font-semibold px-1";

  return (
    <div>
      <div className="mb-4">
        <h3 className="mb-2 text-xl">Abschnittsweiser Vergleich</h3>
        <table className="text-xs">
          <thead>
            <tr className="text-center">
              <th colSpan={5} className={thTopClass}>
                Fahrtabschnitt
              </th>
              <th colSpan={5} className={cn(thTopClass, "bg-red-100")}>
                Zähldaten
              </th>
              <th colSpan={4} className={cn(thTopClass, "bg-blue-100")}>
                RSL-Prognose (Reisende)
              </th>
              <th
                colSpan={2}
                className={cn(
                  thTopClass,
                  "bg-gradient-to-r from-yellow-100 to-yellow-200",
                )}
              >
                Vergleich Plan
              </th>
              <th
                colSpan={6}
                className={cn(
                  thTopClass,
                  "bg-gradient-to-r from-green-100 to-green-300",
                )}
              >
                Vergleich Prognose
              </th>
            </tr>
            <tr className="text-center">
              <th className={thClass} colSpan={2}>
                Von
              </th>
              <th className={thClass} colSpan={2}>
                Nach
              </th>
              <th className={thClass}>Dauer</th>
              <th
                className={cn(thClass, "bg-red-100")}
                title="Anzahl der Reisende in den Zähldaten (alle Einträge)"
              >
                Eintr.
              </th>
              <th
                className={cn(thClass, "bg-red-100")}
                title="Reisende mit Kontrolle oder Check-In"
              >
                Min
              </th>
              <th
                className={cn(thClass, "bg-red-100")}
                title="Durchschnitt aus Min und Max"
              >
                Avg
              </th>
              <th
                className={cn(thClass, "bg-red-100")}
                title="Kontrolliert + Nicht abgedeckt + aus Punktdaten"
              >
                Max
              </th>
              <th
                className={cn(thClass, "bg-red-100")}
                title="Anzahl kontrollierte Einträge in diesem Abschnitt"
              >
                K.i.A.
              </th>
              <th className={cn(thClass, "bg-blue-100")}>Plan</th>
              <th className={cn(thClass, "bg-blue-100")}>5 %</th>
              <th className={cn(thClass, "bg-blue-100")}>50 %</th>
              <th className={cn(thClass, "bg-blue-100")}>95 %</th>
              <th colSpan={2} className={cn(thClass, "bg-yellow-100")}>
                Avg
              </th>
              <th colSpan={2} className={cn(thClass, "bg-green-100")}>
                5 % / Min
              </th>
              <th colSpan={2} className={cn(thClass, "bg-green-200")}>
                50 % / Avg
              </th>
              <th colSpan={2} className={cn(thClass, "bg-green-300")}>
                95 % / Max
              </th>
              <th></th>
            </tr>
          </thead>
          <tbody>{sectionSummary}</tbody>
        </table>
        {sectionsWithChecks != 0 && (
          <div className="mt-4">
            <table>
              <thead>
                <tr className="text-center">
                  <th className={cn(thClass, "text-left")}>Vergleich</th>
                  <th className={cn(thClass)}>MAE</th>
                  <th className={cn(thClass)}>MSE</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="pr-2">Plan vs. Kontrolliert</td>
                  <td className="px-2">{formatNumber(Math.round(maePlan))}</td>
                  <td className="px-2">{formatNumber(Math.round(msePlan))}</td>
                </tr>
                <tr>
                  <td className="pr-2">50 % Prognose vs. Kontrolliert</td>
                  <td className="px-2">{formatNumber(Math.round(maeQ50))}</td>
                  <td className="px-2">{formatNumber(Math.round(mseQ50))}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
      {sectionDetails}
    </div>
  );
}

const REF_COLORS = [
  "bg-red-700",
  "bg-blue-700",
  "bg-green-700",
  "bg-amber-700",
  "bg-teal-700",
  "bg-purple-700",
  "bg-pink-700",
  "bg-orange-700",
  "bg-lime-700",
];

const TRIP_COLORS = REF_COLORS.toReversed();

interface OrderCheckDataCardProps {
  entryRef: number;
  orderId: string;
}

function OrderCheckDataCard({ entryRef, orderId }: OrderCheckDataCardProps) {
  const request: PaxMonCheckDataByOrderRequest = { order_id: orderId };
  const { data, isPending, error } = useQuery({
    queryKey: queryKeys.checkDataByOrder(request),
    queryFn: () => sendPaxMonCheckDataByOrderRequest(request),
  });

  if (!data) {
    if (isPending) {
      return <div>Auftragsdaten werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Auftragsdaten:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
        </div>
      );
    }
  }

  const entries = data.entries.sort((a, b) => {
    const check_time = a.check_min_time - b.check_min_time;
    if (a.check_min_time != 0 && b.check_min_time != 0 && check_time != 0) {
      return check_time;
    }
    const leg_time = a.leg_start_time - b.leg_start_time;
    if (a.leg_start_time != 0 && b.leg_start_time != 0 && leg_time != 0) {
      return leg_time;
    }
    const start_time =
      a.schedule_train_start_time - b.schedule_train_start_time;
    if (start_time != 0) {
      return start_time;
    }
    return a.ref - b.ref;
  });

  const refs: number[] = [];
  const trips: string[] = [];

  for (const entry of entries) {
    if (!refs.includes(entry.ref)) {
      refs.push(entry.ref);
    }
    if (entry.trip_id != "" && !trips.includes(entry.trip_id)) {
      trips.push(entry.trip_id);
    }
  }

  return (
    <div className="text-left text-base">
      <div className="mb-4 flex gap-4 px-2 font-semibold">
        <span>
          Auftrag {orderId}: {entries.length}{" "}
          {entries.length == 1 ? "Eintrag" : "Einträge"}
        </span>
      </div>
      <div className="flex flex-col gap-2">
        {entries.map((entry) => (
          <div
            key={entry.ref}
            className={cn(
              "flex flex-col gap-1 p-2",
              entry.ref == entryRef && "bg-yellow-100",
            )}
          >
            <div className="flex items-center gap-2 border-b border-b-gray-500">
              <span className="flex min-w-36 items-center gap-2">
                <ColorDot index={refs.indexOf(entry.ref)} colors={REF_COLORS} />
                <span>PK {entry.ref}</span>
              </span>
              <span className="min-w-5">
                {entry.ref == entryRef && <Asterisk className="h-5 w-5" />}
              </span>
              <span className="min-w-20">
                {entry.category} {entry.train_nr}
              </span>
              <span className="min-w-40">
                {longCheckType(entry.check_type)}
              </span>
              <span className="min-w-28">
                <Direction direction={entry.direction} />
              </span>
              <span>
                Zug Start: {optionalTime(entry.schedule_train_start_time)}
              </span>
              <div
                className="flex grow items-center justify-end gap-2"
                title={entry.trip_id}
              >
                <span>Trip</span>
                <ColorDot
                  index={trips.indexOf(entry.trip_id)}
                  colors={TRIP_COLORS}
                />
              </div>
            </div>
            <div className="flex items-center gap-4">
              <BoolProperty value={entry.planned_train}>Planmäßig</BoolProperty>
              <BoolProperty value={entry.checked_in_train}>
                Kontrolliert
              </BoolProperty>
              <BoolProperty value={entry.canceled}>Ausfall</BoolProperty>
              <div>Status {longLegStatus(entry.leg_status)}</div>
            </div>
            {entry.leg_start_station && (
              <div className="flex items-center gap-4">
                <span>Reiseabschnitt:</span>
                <span>{optionalStation(entry.leg_start_station)}</span>
                <span>{optionalTime(entry.leg_start_time)}</span>
                <span>
                  <ArrowRight className="h-5 w-5" />
                </span>
                <span>{optionalStation(entry.leg_destination_station)}</span>
                <span>{optionalTime(entry.leg_destination_time)}</span>
              </div>
            )}
            {entry.checkin_start_station && (
              <div className="flex items-center gap-4">
                <span>Check-in:</span>
                <span>{optionalStation(entry.checkin_start_station)}</span>
                <span>
                  <ArrowRight className="h-5 w-5" />
                </span>
                <span>
                  {optionalStation(entry.checkin_destination_station)}
                </span>
              </div>
            )}
            {entry.check_min_time != 0 && (
              <div>
                Kontrolliert: {optionalTime(entry.check_min_time)}
                {entry.check_max_time != entry.check_min_time &&
                  ` – ${optionalTime(entry.check_max_time)}`}
                {` (${entry.check_count}x)`}
              </div>
            )}
            {entry.planned_trip_ref != 0 && (
              <div className="flex items-center gap-2">
                <span>Planzug:</span>
                <ColorDot
                  index={refs.indexOf(entry.planned_trip_ref)}
                  colors={REF_COLORS}
                />
                <span>PK {entry.planned_trip_ref}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

interface DirectionProps {
  direction: PaxMonCheckDirection;
}

function Direction({ direction }: DirectionProps) {
  const wrapperClass = "flex items-center gap-1";
  const iconClass = "h-5 w-5 text-gray-600";

  switch (direction) {
    case "UNKNOWN":
      return (
        <div className={wrapperClass}>
          <HelpCircle className={iconClass} />
          Richtung?
        </div>
      );
    case "OUTWARD":
      return (
        <div className={wrapperClass}>
          <ArrowBigRight className={iconClass} />
          Hinfahrt
        </div>
      );
    case "RETURN":
      return (
        <div className={wrapperClass}>
          <ArrowBigLeft className={iconClass} />
          Rückfahrt
        </div>
      );
  }
}

interface BoolPropertyProps {
  value: boolean;
  children: ReactNode;
}

function BoolProperty({ value, children }: BoolPropertyProps) {
  const iconClass = "h-5 w-5";
  return (
    <div
      className={cn(
        "flex items-center gap-1",
        value ? "text-gray-900" : "text-gray-300",
      )}
    >
      {value ? (
        <CheckCircle2 className={iconClass} />
      ) : (
        <XCircle className={iconClass} />
      )}
      {children}
    </div>
  );
}

interface ColorDotProps {
  index: number;
  colors: string[];
  unknownColor?: string;
}

function ColorDot({
  index,
  colors,
  unknownColor = "bg-gray-700",
}: ColorDotProps) {
  const color = index >= 0 ? colors[index % colors.length] : unknownColor;
  const label = index >= 0 ? `${index + 1}` : "?";
  return (
    <span
      className={cn(
        "flex h-5 w-5 items-center justify-center rounded-full text-xs text-white",
        color,
      )}
    >
      {label}
    </span>
  );
}
