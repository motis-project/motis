import { PencilIcon, TrashIcon } from "@heroicons/react/solid";
import { PrimitiveAtom, atom, useAtom } from "jotai";
import { useAtomCallback, useUpdateAtom } from "jotai/utils";
import { useCallback } from "react";
import { useMutation, useQueryClient } from "react-query";

import { LoadLevel, MeasureWrapper } from "@/api/protocol/motis/paxforecast";
import { PaxMonStatusResponse } from "@/api/protocol/motis/paxmon";

import { sendPaxForecastApplyMeasuresRequest } from "@/api/paxforecast";
import { queryKeys } from "@/api/paxmon";

import {
  MeasureUnion,
  currentEditorMeasureAtom,
  measuresAtom,
  newEmptyMeasure,
  toMeasureWrapper,
} from "@/data/measures";
import {
  SimulationResult,
  selectedSimResultAtom,
  simResultsAtom,
  universeAtom,
} from "@/data/simulation";

import { formatDateTime } from "@/util/dateFormat";
import { isNonNull } from "@/util/typeGuards";

import TripServiceInfoView from "@/components/TripServiceInfoView";

type RemoveFn = (ma: PrimitiveAtom<MeasureUnion>) => void;
type SelectFn = (ma: PrimitiveAtom<MeasureUnion>) => void;

const loadLevels: Record<LoadLevel, string> = {
  Unknown: "unbekannt",
  Low: "gering",
  NoSeats: "keine Sitzplätze",
  Full: "keine Mitfahrmöglichkeit",
};

function MeasureTypeDetailColumn({
  measure,
}: {
  measure: MeasureUnion;
}): JSX.Element {
  switch (measure.type) {
    case "TripLoadInfoMeasure": {
      return (
        <>
          <div className="text-sm font-medium text-gray-900">
            Auslastungsinformation
          </div>
          <div className="text-sm text-gray-500">
            {measure.data.trip ? (
              <>
                <TripServiceInfoView tsi={measure.data.trip} format="Short" />
                <span>
                  {": "}
                  {loadLevels[measure.data.level]}
                </span>
              </>
            ) : (
              <span className="text-db-red-500">Kein Trip gewählt</span>
            )}
          </div>
        </>
      );
    }
    case "TripRecommendationMeasure": {
      return (
        <>
          <div className="text-sm font-medium text-gray-900">Zugempfehlung</div>
          <div className="text-sm text-gray-500">
            {measure.data.recommended_trip ? (
              <TripServiceInfoView
                tsi={measure.data.recommended_trip}
                format="Short"
              />
            ) : (
              <span className="text-db-red-500">Kein Trip gewählt</span>
            )}
          </div>
        </>
      );
    }
    case "RtUpdateMeasure": {
      return (
        <>
          <div className="text-sm font-medium text-gray-900">
            Echtzeitmeldung
          </div>
          <div className="text-sm text-gray-500">
            {measure.data.trip ? (
              <TripServiceInfoView tsi={measure.data.trip} format="Short" />
            ) : (
              <span className="text-db-red-500">Kein Trip gewählt</span>
            )}
          </div>
        </>
      );
    }
    case "Empty": {
      return (
        <>
          <div className="text-sm font-medium text-gray-900">Neue Maßnahme</div>
        </>
      );
    }
  }
}

function MeasureSharedDataColumn({
  measure,
}: {
  measure: MeasureUnion;
}): JSX.Element {
  const shared = measure.shared;
  return (
    <>
      <div className="text-sm text-gray-900">{formatDateTime(shared.time)}</div>
      {shared.recipients.stations.length > 0 && (
        <div className="text-sm text-gray-500">
          {shared.recipients.stations[0].name}
          {shared.recipients.stations.length > 1 &&
            ` (+${shared.recipients.stations.length - 1})`}
        </div>
      )}
      {shared.recipients.trips.length > 0 && (
        <div className="text-sm text-gray-500">
          <TripServiceInfoView
            tsi={shared.recipients.trips[0]}
            format="Short"
          />
          {shared.recipients.trips.length > 1 &&
            ` (+${shared.recipients.trips.length - 1})`}
        </div>
      )}
    </>
  );
}

type MeasureListEntryProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  remove: RemoveFn;
  select: SelectFn;
};

function MeasureListEntry({
  measureAtom,
  remove,
  select,
}: MeasureListEntryProps) {
  const [measure] = useAtom(measureAtom);

  return (
    <tr>
      <td className="px-4 py-3 whitespace-nowrap">
        <MeasureTypeDetailColumn measure={measure} />
      </td>
      <td className="px-4 py-3 whitespace-nowrap">
        <MeasureSharedDataColumn measure={measure} />
      </td>
      <td className="px-4 py-3 whitespace-nowrap text-right text-sm font-medium">
        <button onClick={() => select(measureAtom)} className="p-1">
          <PencilIcon className="h-4 w-4 text-gray-900 hover:text-gray-600" />
          <span className="sr-only">Bearbeiten</span>
        </button>
        <button onClick={() => remove(measureAtom)} className="p-1 ml-2">
          <TrashIcon className="h-4 w-4 text-gray-900 hover:text-gray-600" />
          <span className="sr-only">Löschen</span>
        </button>
      </td>
    </tr>
  );
}

export type MeasureListProps = {
  onSimulationFinished: () => void;
};

function MeasureList({ onSimulationFinished }: MeasureListProps): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [measureAtoms, setMeasureAtoms] = useAtom(measuresAtom);
  const [selectedMeasure, setSelectedMeasure] = useAtom(
    currentEditorMeasureAtom
  );
  const setSimResults = useUpdateAtom(simResultsAtom);
  const setSelectedSimResult = useUpdateAtom(selectedSimResultAtom);

  const applyMeasuresMutation = useMutation(
    (measures: MeasureWrapper[]) =>
      sendPaxForecastApplyMeasuresRequest({
        universe,
        measures,
        replace_existing: true,
        preparation_time: 0,
        include_before_trip_load_info: true,
        include_after_trip_load_info: true,
      }),
    {
      onMutate: () => {
        return { startedAt: new Date() };
      },
      onSuccess: async (data, variables, context) => {
        console.log("measures applied");
        const result: SimulationResult = {
          universe,
          startedAt: context.startedAt,
          finishedAt: new Date(),
          measures: variables,
          response: data,
        };
        const resultAtom = atom(result);
        setSimResults((prev) => {
          return [...prev, resultAtom];
        });
        setSelectedSimResult(resultAtom);
        await queryClient.invalidateQueries(queryKeys.all);
        onSimulationFinished();
      },
      retry: false,
    }
  );

  const applyMeasures = useAtomCallback(
    useCallback(
      (get) => {
        console.log("applyMeasures");
        const atoms = get(measuresAtom);
        console.log(`${atoms.length} atoms`);
        const measures = atoms.map((a) => get(a));
        const measureWrappers = measures
          .map(toMeasureWrapper)
          .filter(isNonNull);
        console.log(measures);
        console.log(JSON.stringify(measureWrappers, null, 2));
        applyMeasuresMutation.mutate(measureWrappers);
      },
      [applyMeasuresMutation]
    )
  );

  const add = () => {
    const systemTime = queryClient.getQueryData<PaxMonStatusResponse>(
      queryKeys.status(universe)
    )?.system_time;
    const currentTime = systemTime ? new Date(systemTime * 1000) : new Date();
    const newMeasure = atom<MeasureUnion>(newEmptyMeasure(currentTime));
    setMeasureAtoms((prev) => [...prev, newMeasure]);
    setSelectedMeasure(newMeasure);
  };

  const remove: RemoveFn = (ma) => {
    setMeasureAtoms((prev) => prev.filter((e) => e !== ma));
    if (selectedMeasure === ma) {
      setSelectedMeasure(null);
    }
  };

  const clear = () => {
    setMeasureAtoms([]);
    setSelectedMeasure(null);
  };

  const applyEnabled =
    universe != 0 &&
    measureAtoms.length > 0 &&
    !applyMeasuresMutation.isLoading;

  return (
    <div className="flex flex-col gap-2 h-full overflow-hidden">
      <div className="flex justify-between">
        <span className="text-xl">Maßnahmen</span>
        <div className="flex gap-2">
          <button
            onClick={add}
            className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-sm rounded"
          >
            Maßnahme hinzufügen
          </button>
          <button
            onClick={clear}
            className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-sm rounded"
          >
            Alle löschen
          </button>
        </div>
      </div>
      <div className="overflow-y-auto grow pr-2">
        {measureAtoms.length > 0 ? (
          <div className="shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th
                    scope="col"
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    Maßnahme
                  </th>
                  <th
                    scope="col"
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    Zeit &amp; Ort
                  </th>
                  <th scope="col" className="relative px-4 py-3">
                    <span className="sr-only">Bearbeiten</span>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {measureAtoms.map((ma) => (
                  <MeasureListEntry
                    key={`${ma}`}
                    measureAtom={ma}
                    remove={remove}
                    select={setSelectedMeasure}
                  />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-db-cool-gray-600">
            Noch keine Maßnahmen hinzugefügt.
          </div>
        )}
      </div>
      {universe === 0 && (
        <div>
          Für die Simulation von Maßnahmen muss zuerst ein Paralleluniversum
          angelegt und ausgewählt werden. Die Simulation von Maßnahmen im
          Hauptuniversum ist nicht möglich.
        </div>
      )}
      <div>
        <button
          onClick={applyMeasures}
          disabled={!applyEnabled}
          className={`w-full p-3 rounded ${
            applyMeasuresMutation.isLoading
              ? "bg-db-red-300 text-db-red-100 cursor-wait"
              : applyEnabled
              ? "bg-db-red-500 hover:bg-db-red-600 text-white"
              : "bg-db-red-300 text-db-red-100 cursor-not-allowed"
          }`}
        >
          Maßnahmen simulieren
        </button>
      </div>
    </div>
  );
}

export default MeasureList;
