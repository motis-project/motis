import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, SelectorIcon } from "@heroicons/react/solid";
import { differenceInMilliseconds } from "date-fns";
import { PrimitiveAtom, useAtom } from "jotai";
import { useUpdateAtom } from "jotai/utils";
import { Fragment, memo } from "react";
import { Virtuoso } from "react-virtuoso";

import { PaxMonUpdatedTrip } from "@/api/protocol/motis/paxmon";

import { formatMiliseconds, formatNumber } from "@/data/numberFormat";
import { selectedTripAtom } from "@/data/selectedTrip";
import {
  SimulationResult,
  selectedSimResultAtom,
  simResultsAtom,
} from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatDateTime } from "@/util/dateFormat";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";
import TripServiceInfoView from "@/components/TripServiceInfoView";

type SimResultsListEntryProps = {
  simResultAtom: PrimitiveAtom<SimulationResult>;
};

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
    selectedSimResultAtom
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
              <SelectorIcon
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
                    classNames(
                      "cursor-default select-none relative py-2 pl-10 pr-4",
                      active ? "text-amber-900 bg-amber-100" : "text-gray-900"
                    )
                  }
                  value={resultAtom}
                >
                  {({ selected, active }) => (
                    <>
                      <span
                        className={classNames(
                          "block truncate",
                          selected ? "font-medium" : "font-normal"
                        )}
                      >
                        <SimResultsListEntry simResultAtom={resultAtom} />
                      </span>
                      {selected ? (
                        <span
                          className={classNames(
                            "absolute inset-y-0 left-0 flex items-center pl-3",
                            active ? "text-amber-600" : "text-amber-600"
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

type SimResultDetailsProps = {
  simResultAtom: PrimitiveAtom<SimulationResult>;
};

function SimResultDetails({
  simResultAtom,
}: SimResultDetailsProps): JSX.Element {
  const [simResult] = useAtom(simResultAtom);

  const r = simResult.response;
  const duration = differenceInMilliseconds(
    simResult.finishedAt,
    simResult.startedAt
  );

  const MemoizedUpdatedTrip = memo(function MemoizedUpdatedTripWrapper({
    index,
  }: {
    index: number;
  }) {
    return <UpdatedTrip ut={r.updates.updated_trips[index]} />;
  });

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
  ]
    .map(({ duration, label }) => `${formatMiliseconds(duration)} ${label}`)
    .join("\n");

  return (
    <>
      <div>
        <div className="my-3 text-lg font-semibold">Statistiken:</div>
        <ul>
          <li>
            Betroffene Reisendengruppen:{" "}
            {formatNumber(r.stats.total_affected_groups)}
          </li>
          <li>
            Alternativensuchen:{" "}
            {formatNumber(r.stats.total_alternative_routings)} (
            {formatNumber(r.stats.total_alternatives_found)} Ergebnisse,{" "}
            {formatMiliseconds(r.stats.t_find_alternatives)})
          </li>
          <li title={runtimeStats}>
            Simulationsdauer insgesamt: {formatMiliseconds(duration)}
          </li>
        </ul>
        <div className="my-3 text-lg font-semibold">
          {formatNumber(r.updates.updated_trip_count)} betroffene Züge:
        </div>
      </div>
      <div className="grow">
        <Virtuoso
          data={r.updates.updated_trips}
          overscan={200}
          itemContent={(index) => <MemoizedUpdatedTrip index={index} />}
        />
      </div>
    </>
  );
}

type UpdatedTripProps = {
  ut: PaxMonUpdatedTrip;
};

function UpdatedTrip({ ut }: UpdatedTripProps) {
  const setSelectedTrip = useUpdateAtom(selectedTripAtom);

  return (
    <div
      className="flex flex-col gap-2 py-3 pr-2 cursor-pointer"
      onClick={() => setSelectedTrip(ut.tsi)}
    >
      <TripServiceInfoView tsi={ut.tsi} format="Long" />
      <ul>
        <li>
          Reisende über Kapazität: {ut.critical_info_before.max_excess_pax}
          {" → "}
          {ut.critical_info_after.max_excess_pax} max. /{" "}
          {ut.critical_info_before.cumulative_excess_pax} {" → "}
          {ut.critical_info_after.cumulative_excess_pax} gesamt
        </li>
        <li>
          Kritische Abschnitte: {ut.critical_info_before.critical_sections}
          {" → "}
          {ut.critical_info_after.critical_sections}
        </li>
        <li>
          Reisende: Avg: -{Math.round(ut.removed_mean_pax)} +
          {Math.round(ut.added_mean_pax)} / Max: -{ut.removed_max_pax} +
          {ut.added_max_pax}
        </li>
      </ul>
      <MiniTripLoadGraph edges={ut.before_edges} />
      <MiniTripLoadGraph edges={ut.after_edges} />
    </div>
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
