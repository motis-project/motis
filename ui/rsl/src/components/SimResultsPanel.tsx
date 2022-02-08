import { differenceInMilliseconds } from "date-fns";
import { PrimitiveAtom, useAtom } from "jotai";
import { useState } from "react";

import { SimulationResult, simResultsAtom } from "../data/simulation";

import { formatDateTime } from "../util/dateFormat";

import MiniTripLoadGraph from "./MiniTripLoadGraph";
import TripServiceInfoView from "./TripServiceInfoView";

type SimResultsListEntryProps = {
  simResultAtom: PrimitiveAtom<SimulationResult>;
  onSelectSim: (sim: PrimitiveAtom<SimulationResult>) => void;
};

function SimResultsListEntry({
  simResultAtom,
  onSelectSim,
}: SimResultsListEntryProps): JSX.Element {
  const [simResult] = useAtom(simResultAtom);

  return (
    <div className="cursor-pointer" onClick={() => onSelectSim(simResultAtom)}>
      Universum #{simResult.universe}, {formatDateTime(simResult.startedAt)}
    </div>
  );
}

type SimResultsListProps = {
  onSelectSim: (sim: PrimitiveAtom<SimulationResult>) => void;
};

function SimResultsList({ onSelectSim }: SimResultsListProps): JSX.Element {
  const [simResultsList] = useAtom(simResultsAtom);

  return (
    <div>
      <div className="my-4 text-lg font-semibold">Simulationsergebnisse:</div>
      {simResultsList.map((a) => (
        <SimResultsListEntry
          key={a.toString()}
          simResultAtom={a}
          onSelectSim={onSelectSim}
        />
      ))}
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

  // TODO(pablo): sort updated trips (by number of changes, critical sections...?)

  return (
    <div>
      <div>
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
      </div>
      <div>
        <div className="my-4 text-lg font-semibold">Betroffene Trips:</div>
        <div className="flex flex-col gap-4">
          {r.updates.updated_trips.map((ut) => (
            <div key={JSON.stringify(ut.tsi)} className="flex flex-col gap-2">
              <TripServiceInfoView tsi={ut.tsi} format="Long" />
              <div>
                Entfernt: {Math.round(ut.removed_mean_pax)} avg /{" "}
                {ut.removed_max_pax} max Reisende
                <br />
                Hinzugefügt: {Math.round(ut.added_mean_pax)} avg /{" "}
                {ut.added_max_pax} max Reisende
              </div>
              <MiniTripLoadGraph edges={ut.before_edges} />
              <MiniTripLoadGraph edges={ut.after_edges} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SimResultsPanel(): JSX.Element {
  const [selectedSim, setSelectedSim] =
    useState<PrimitiveAtom<SimulationResult>>();

  return (
    <div>
      <SimResultsList onSelectSim={setSelectedSim} />
      {selectedSim ? <SimResultDetails simResultAtom={selectedSim} /> : null}
    </div>
  );
}

export default SimResultsPanel;
