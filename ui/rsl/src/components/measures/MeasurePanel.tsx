import { PrimitiveAtom, atom, useAtom } from "jotai";
import { useAtomCallback, useUpdateAtom } from "jotai/utils";
import { useCallback } from "react";
import { useMutation, useQueryClient } from "react-query";

import { sendPaxForecastApplyMeasuresRequest } from "../../api/paxforecast";
import { queryKeys } from "../../api/paxmon";
import { MeasureWrapper } from "../../api/protocol/motis/paxforecast";
import { PaxMonStatusResponse } from "../../api/protocol/motis/paxmon";

import {
  MeasureUnion,
  currentEditorMeasureAtom,
  measuresAtom,
  newEmptyMeasure,
  toMeasureWrapper,
} from "../../data/measures";
import {
  SimulationResult,
  simResultsAtom,
  universeAtom,
} from "../../data/simulation";

import { isNonNull } from "../../util/typeGuards";

import MeasureEditor from "./MeasureEditor";

type RemoveFn = (ma: PrimitiveAtom<MeasureUnion>) => void;
type SelectFn = (ma: PrimitiveAtom<MeasureUnion>) => void;

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
    <div>
      {`${measureAtom}`}: {measure.type}
      <button
        onClick={() => select(measureAtom)}
        className="ml-3 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
      >
        Bearbeiten
      </button>
      <button
        onClick={() => remove(measureAtom)}
        className="ml-3 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
      >
        Entfernen
      </button>
    </div>
  );
}

function MeasureList() {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [measureAtoms, setMeasureAtoms] = useAtom(measuresAtom);
  const [selectedMeasure, setSelectedMeasure] = useAtom(
    currentEditorMeasureAtom
  );
  const setSimResults = useUpdateAtom(simResultsAtom);

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
        setSimResults((prev) => {
          return [...prev, atom(result)];
        });
        await queryClient.invalidateQueries(queryKeys.trip());
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

  const applyEnabled = universe != 0;

  return (
    <div className="flex flex-col gap-2">
      <div>
        {measureAtoms.map((ma) => (
          <MeasureListEntry
            key={`${ma}`}
            measureAtom={ma}
            remove={remove}
            select={setSelectedMeasure}
          />
        ))}
      </div>
      <div>
        <button
          onClick={add}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
        >
          Neue Maßnahme hinzufügen
        </button>
      </div>
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

function MeasurePanel(): JSX.Element {
  // TODO: modal panel: only show one of {list (+ apply button), new measure, edit measure}
  const [currentMeasureAtom] = useAtom(currentEditorMeasureAtom);
  return (
    <div>
      <MeasureList />
      {currentMeasureAtom ? (
        <MeasureEditor measureAtom={currentMeasureAtom} />
      ) : null}
    </div>
  );
}

export default MeasurePanel;
