import { Listbox, Transition } from "@headlessui/react";
import {
  ArrowSmallDownIcon,
  CheckCircleIcon,
  CheckIcon,
  ChevronUpDownIcon,
  ExclamationCircleIcon,
} from "@heroicons/react/20/solid";
import { differenceInMilliseconds } from "date-fns";
import { PrimitiveAtom, useAtom } from "jotai";
import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import { Virtuoso } from "react-virtuoso";

import { PaxMonUpdatedTrip } from "@/api/protocol/motis/paxmon";

import { formatMiliseconds, formatNumber } from "@/data/numberFormat";
import {
  SimulationResult,
  selectedSimResultAtom,
  simResultsAtom,
} from "@/data/simulation";

import { formatDateTime } from "@/util/dateFormat";

import MiniTripLoadGraph from "@/components/trips/MiniTripLoadGraph";

import { cn } from "@/lib/utils";

interface SimResultsListEntryProps {
  simResultAtom: PrimitiveAtom<SimulationResult>;
}

function SimResultsListEntry({
  simResultAtom,
}: SimResultsListEntryProps): JSX.Element {
  const [simResult] = useAtom(simResultAtom);

  return (
    <span>
      {formatDateTime(simResult.startedAt)} &ndash; Universum #
      {simResult.universe}
    </span>
  );
}

function SimResultsList(): JSX.Element {
  const [simResultsList] = useAtom(simResultsAtom);
  const [selectedSimResult, setSelectedSimResult] = useAtom(
    selectedSimResultAtom,
  );

  return (
    <div>
      <Listbox value={selectedSimResult} onChange={setSelectedSimResult}>
        <div className="relative mt-1">
          <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
            <span className="block truncate">
              {selectedSimResult ? (
                <SimResultsListEntry simResultAtom={selectedSimResult} />
              ) : (
                <span className="text-db-cool-gray-700">
                  Simulation auswählen...
                </span>
              )}
            </span>
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
              {simResultsList.map((resultAtom, resultIdx) => (
                <Listbox.Option
                  key={resultIdx}
                  className={({ active }) =>
                    cn(
                      "cursor-default select-none relative py-2 pl-10 pr-4",
                      active ? "text-amber-900 bg-amber-100" : "text-gray-900",
                    )
                  }
                  value={resultAtom}
                >
                  {({ selected, active }) => (
                    <>
                      <span
                        className={cn(
                          "block truncate",
                          selected ? "font-medium" : "font-normal",
                        )}
                      >
                        <SimResultsListEntry simResultAtom={resultAtom} />
                      </span>
                      {selected ? (
                        <span
                          className={cn(
                            "absolute inset-y-0 left-0 flex items-center pl-3",
                            active ? "text-amber-600" : "text-amber-600",
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
    </div>
  );
}

function hasCritChange(ut: PaxMonUpdatedTrip): boolean {
  return ut.newly_critical_sections > 0 || ut.no_longer_critical_sections > 0;
}

interface SimResultDetailsProps {
  simResultAtom: PrimitiveAtom<SimulationResult>;
}

function SimResultDetails({
  simResultAtom,
}: SimResultDetailsProps): JSX.Element {
  const [simResult] = useAtom(simResultAtom);
  const [critChangeOnly, setCritChangeOnly] = useState(false);

  const r = simResult.response;
  const duration = differenceInMilliseconds(
    simResult.finishedAt,
    simResult.startedAt,
  );

  const trips = critChangeOnly
    ? r.updates.updated_trips.filter(hasCritChange)
    : r.updates.updated_trips;

  const runtimeStats = [
    { duration: r.stats.t_rt_updates, label: "Einspielen der Echtzeitupdates" },
    {
      duration: r.stats.t_get_affected_groups,
      label: "Bestimmung betroffener Gruppen",
    },
    { duration: r.stats.t_find_alternatives, label: "Alternativensuche" },
    { duration: r.stats.t_behavior_simulation, label: "Verhaltenssimulation" },
    {
      duration: r.stats.t_update_groups,
      label: "Aktualisierung der Reisendengruppen",
    },
    { duration: r.stats.t_update_tracker, label: "Statistiken zu Änderungen" },
    {
      duration: r.stats.t_update_capacities,
      label: "Aktualisierung der Kapazitäten",
    },
  ]
    .map(({ duration, label }) => `${formatMiliseconds(duration)} ${label}`)
    .join("\n");

  return (
    <>
      <div className="mt-2">
        <div title={runtimeStats}>
          Simulationsdauer insgesamt: {formatMiliseconds(duration)}
        </div>
        <div
          title={`Betroffene Reisendengruppen: ${formatNumber(
            r.updates.updated_group_count,
          )} (${formatNumber(r.updates.updated_group_route_count)} Routen)`}
        >
          Betroffende Reisende: {formatNumber(r.updates.updated_pax_count)}
        </div>
        <div className="mt-1 font-semibold">
          Nachfragebeeinflussende Maßnahmen
        </div>
        <div className="ml-3">
          <div>
            Betroffene Reisendengruppen:{" "}
            {formatNumber(r.stats.total_affected_groups)}
          </div>
          <div>
            Alternativensuchen:{" "}
            {formatNumber(r.stats.total_alternative_routings)} (
            {formatNumber(r.stats.total_alternatives_found)} Ergebnisse,{" "}
            {formatMiliseconds(r.stats.t_find_alternatives)})
          </div>
        </div>
        <div>
          <div className="mt-1 font-semibold">
            Angebotsbeeinflussende Maßnahmen
          </div>
          <div className="ml-3">
            {`Gebrochene Reiseketten: ${formatNumber(
              r.stats.group_routes_broken,
            )}`}
          </div>
        </div>
        <label className="mt-2 flex items-center gap-2">
          <input
            type="checkbox"
            className="rounded border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-offset-0 focus:ring-blue-200 focus:ring-opacity-50"
            checked={critChangeOnly}
            onChange={() => setCritChangeOnly((b) => !b)}
          />
          Nur Züge mit Änderungen der kritischen Abschnitte anzeigen
        </label>
        <div className="my-3 text-lg font-semibold">
          <span>
            {formatNumber(r.updates.updated_trip_count)} betroffene Züge:
          </span>
        </div>
      </div>
      <div className="grow">
        <Virtuoso
          data={trips}
          overscan={200}
          itemContent={(index) => <UpdatedTrip ut={trips[index]} />}
        />
      </div>
    </>
  );
}

interface UpdatedTripProps {
  ut: PaxMonUpdatedTrip;
}

function UpdatedTrip({ ut }: UpdatedTripProps) {
  const category = ut.tsi.service_infos[0]?.category ?? "";
  const trainNr = ut.tsi.service_infos[0]?.train_nr ?? ut.tsi.trip.train_nr;

  return (
    <Link
      to={`/trips/${encodeURIComponent(JSON.stringify(ut.tsi.trip))}`}
      className="block pb-3 pr-1"
    >
      <div className="p-1 flex flex-col gap-2 rounded bg-db-cool-gray-100">
        <div className="flex gap-4 pb-1">
          <div className="flex flex-col">
            <div className="text-sm text-center">{category}</div>
            <div className="text-xl font-semibold">{trainNr}</div>
          </div>
          <div className="grow flex flex-col truncate">
            <div className="flex justify-between">
              <div className="truncate">{ut.tsi.primary_station.name}</div>
              <div>{formatDateTime(ut.tsi.trip.time)}</div>
            </div>
            <div className="flex justify-between">
              <div className="truncate">{ut.tsi.secondary_station.name}</div>
              <div>{formatDateTime(ut.tsi.trip.target_time)}</div>
            </div>
          </div>
        </div>
        <ul>
          {ut.rerouted && (
            <li className="text-purple-700">
              Zugverlauf durch Echtzeitupdates geändert
            </li>
          )}
          {ut.newly_critical_sections > 0 && (
            <li className="flex items-center gap-1 text-red-700">
              <ExclamationCircleIcon className="w-5 h-5" />
              {ut.newly_critical_sections > 1
                ? `${ut.newly_critical_sections} neue kritische Abschnitte`
                : "Ein neuer kritischer Abschnitt"}
            </li>
          )}
          {ut.no_longer_critical_sections > 0 && (
            <li className="flex items-center gap-1 text-green-700">
              <CheckCircleIcon className="w-5 h-5" />
              {ut.no_longer_critical_sections > 1
                ? `Auslastung auf ${ut.no_longer_critical_sections} Abschnitten nicht mehr kritisch`
                : "Auslastung auf einem Abschnitt nicht mehr kritisch"}
            </li>
          )}
          {ut.max_pax_increase + ut.max_pax_decrease != 0 && (
            <li>
              Größte Änderung:
              {ut.max_pax_increase > ut.max_pax_decrease
                ? ` +${ut.max_pax_increase} `
                : ` -${ut.max_pax_decrease} `}
              Reisende
            </li>
          )}
        </ul>
        <div className="flex flex-col items-center gap-1">
          <MiniTripLoadGraph edges={ut.before_edges} />
          <ArrowSmallDownIcon className="w-5 h-5 fill-gray-500" />
          <MiniTripLoadGraph edges={ut.after_edges} />
        </div>
      </div>
    </Link>
  );
}

function SimResultsPanel(): JSX.Element {
  const [selectedSimResult] = useAtom(selectedSimResultAtom);

  return (
    <div className="h-full flex flex-col">
      <SimResultsList />
      {selectedSimResult ? (
        <SimResultDetails simResultAtom={selectedSimResult} />
      ) : null}
    </div>
  );
}

export default SimResultsPanel;
