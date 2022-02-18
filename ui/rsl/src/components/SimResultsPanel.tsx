import { differenceInMilliseconds } from "date-fns";
import { PrimitiveAtom, useAtom } from "jotai";
import { memo, useState } from "react";
import { Virtuoso } from "react-virtuoso";

import { PaxMonUpdatedTrip } from "@/api/protocol/motis/paxmon";

import { SimulationResult, simResultsAtom } from "@/data/simulation";

import { formatDateTime } from "@/util/dateFormat";
import useRenderCount from "@/util/useRenderCount";

import MiniTripLoadGraph from "@/components/MiniTripLoadGraph";
import TripServiceInfoView from "@/components/TripServiceInfoView";

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
      <div className="mb-4 text-lg font-semibold">Simulationsergebnisse:</div>
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
      <SimResultsList onSelectSim={setSelectedSim} />
      {selectedSim ? <SimResultDetails simResultAtom={selectedSim} /> : null}
    </div>
  );
}

export default SimResultsPanel;
