import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, SelectorIcon } from "@heroicons/react/solid";
import { differenceInMilliseconds } from "date-fns";
import { PrimitiveAtom, useAtom } from "jotai";
import { Fragment, memo, useState } from "react";
import { Virtuoso } from "react-virtuoso";

import { PaxMonUpdatedTrip } from "@/api/protocol/motis/paxmon";

import { SimulationResult, simResultsAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatDateTime } from "@/util/dateFormat";
import useRenderCount from "@/util/useRenderCount";

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

type SimResultsListProps = {
  selectedSim: PrimitiveAtom<SimulationResult> | undefined;
  onSelectSim: (sim: PrimitiveAtom<SimulationResult>) => void;
};

function SimResultsList({
  selectedSim,
  onSelectSim,
}: SimResultsListProps): JSX.Element {
  const [simResultsList] = useAtom(simResultsAtom);

  return (
    <div>
      <div className="">
        <Listbox value={selectedSim} onChange={onSelectSim}>
          <div className="relative mt-1">
            <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white rounded-lg shadow-md cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500 sm:text-sm">
              <span className="block truncate">
                {selectedSim && (
                  <SimResultsListEntry simResultAtom={selectedSim} />
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
              <Listbox.Options className="absolute w-full py-1 mt-1 overflow-auto text-base bg-white rounded-md shadow-lg max-h-60 ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
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
  const renderCount = useRenderCount();

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

  return (
    <>
      <div>
        <div>SimResultDetails Render Count: {renderCount}</div>
        <div className="my-4 text-lg font-semibold">Statistiken:</div>
        <ul>
          <li>
            Anzahl Maßnahmen: {r.stats.total_measures_applied} (
            {r.stats.measure_time_points} Zeitpunkte)
          </li>
          <li>Betroffene Gruppen: {r.stats.total_affected_groups}</li>
          <li>
            Alternativensuchen: {r.stats.total_alternative_routings} (
            {r.stats.total_alternatives_found} Ergebnisse)
          </li>
          <li>Simulationsdauer: {duration} ms</li>
          <li>Gruppen entfernt: {r.updates.removed_group_count}</li>
          <li>Gruppen hinzugefügt: {r.updates.added_group_count}</li>
          <li>Gruppen wiederverwendet: {r.updates.reused_group_count}</li>
          <li>Trips aktualisiert: {r.updates.updated_trip_count}</li>
        </ul>
        <div className="my-4 text-lg font-semibold">Betroffene Züge:</div>
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
  return (
    <div className="flex flex-col gap-2 py-3 pr-2">
      <TripServiceInfoView tsi={ut.tsi} format="Long" />
      <div>
        Entfernt: {Math.round(ut.removed_mean_pax)} avg / {ut.removed_max_pax}{" "}
        max Reisende
        <br />
        Hinzugefügt: {Math.round(ut.added_mean_pax)} avg / {ut.added_max_pax}{" "}
        max Reisende
        <br />
        Kritische Abschnitte: {ut.critical_info_before.critical_sections}
        {" → "}
        {ut.critical_info_after.critical_sections}
        <br />
        Max. Reisende über Kapazität: {ut.critical_info_before.max_excess_pax}
        {" → "}
        {ut.critical_info_after.max_excess_pax}
        <br />
        Kum. Reisende über Kapazität:{" "}
        {ut.critical_info_before.cumulative_excess_pax} {" → "}
        {ut.critical_info_after.cumulative_excess_pax}
        <br />
      </div>
      <MiniTripLoadGraph edges={ut.before_edges} />
      <MiniTripLoadGraph edges={ut.after_edges} />
    </div>
  );
}

function SimResultsPanel(): JSX.Element {
  const [selectedSim, setSelectedSim] =
    useState<PrimitiveAtom<SimulationResult>>();

  return (
    <div className="h-full flex flex-col">
      <SimResultsList selectedSim={selectedSim} onSelectSim={setSelectedSim} />
      {selectedSim ? <SimResultDetails simResultAtom={selectedSim} /> : null}
    </div>
  );
}

export default SimResultsPanel;
