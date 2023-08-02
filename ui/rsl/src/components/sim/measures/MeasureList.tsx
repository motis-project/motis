import { PencilIcon, TrashIcon } from "@heroicons/react/20/solid";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PrimitiveAtom, atom, useAtom, useSetAtom } from "jotai";
import { useAtomCallback } from "jotai/utils";
import { useCallback } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel, MeasureWrapper } from "@/api/protocol/motis/paxforecast";
import { PaxMonStatusResponse } from "@/api/protocol/motis/paxmon";

import { sendPaxForecastApplyMeasuresRequest } from "@/api/paxforecast";
import { queryKeys } from "@/api/paxmon";

import {
  MeasureUnion,
  UiMeasureType,
  currentEditorMeasureAtom,
  measureNeedsRecipients,
  measuresAtom,
  newEmptyMeasure,
  toMeasureWrapper,
} from "@/data/measures";
import { universeAtom } from "@/data/multiverse";
import {
  SimulationResult,
  selectedSimResultAtom,
  simResultsAtom,
} from "@/data/simulation";

import { formatDateTime } from "@/util/dateFormat";
import { loadLevelInfos } from "@/util/loadLevelInfos";
import { isNonNull } from "@/util/typeGuards";

import TripServiceInfoView from "@/components/TripServiceInfoView";

import { cn } from "@/lib/utils";

type RemoveFn = (ma: PrimitiveAtom<MeasureUnion>) => void;
type SelectFn = (ma: PrimitiveAtom<MeasureUnion>) => void;

const measureTypeTexts: Record<UiMeasureType, string> = {
  TripLoadInfoMeasure: "Auslastungsinformation",
  TripRecommendationMeasure: "Zugempfehlung",
  TripLoadRecommendationMeasure: "Alternativenempfehlung",
  RtUpdateMeasure: "Echtzeitmeldung",
  RtCancelMeasure: "(Teil-)Ausfall",
  UpdateCapacitiesMeasure: "Kapazitätsänderung",
  OverrideCapacityMeasure: "Kapazitätsänderung",
  Empty: "Neue Maßnahme",
};

function MeasureTypeDetail({
  measure,
}: {
  measure: MeasureUnion;
}): JSX.Element {
  switch (measure.type) {
    case "TripLoadInfoMeasure": {
      return (
        <div className="text-sm text-gray-500">
          <TripWithLoadLevel
            tsi={measure.data.trip}
            level={measure.data.level}
            placeholder="Kein Zug ausgewählt"
          />
        </div>
      );
    }
    case "TripRecommendationMeasure": {
      return (
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
      );
    }
    case "TripLoadRecommendationMeasure": {
      return (
        <div className="text-sm text-gray-800">
          <div
            className={cn(
              "truncate",
              !measure.data.planned_destination && "text-db-red-500",
            )}
          >
            {measure.data.planned_destination
              ? `→ ${measure.data.planned_destination.name}`
              : "Keine Richtung ausgewählt"}
          </div>
          <TripWithLoadLevel
            tsi={measure.data.full_trip.trip}
            level={measure.data.full_trip.level}
            placeholder="Kein überfüllter Zug ausgewählt"
          />
          {measure.data.recommended_trips.map((tll, idx) => (
            <TripWithLoadLevel key={idx} tsi={tll.trip} level={tll.level} />
          ))}
        </div>
      );
    }
    case "RtUpdateMeasure": {
      return (
        <div className="text-sm text-gray-500">
          {measure.data.trip ? (
            <TripServiceInfoView tsi={measure.data.trip} format="Short" />
          ) : (
            <span className="text-db-red-500">Kein Trip gewählt</span>
          )}
        </div>
      );
    }
    case "RtCancelMeasure": {
      const canceled = measure.data.canceled_stops.filter((c) => c).length;
      const allCanceled = canceled == measure.data.canceled_stops.length;
      return (
        <div className="text-sm text-gray-500">
          {measure.data.trip ? (
            <>
              <TripServiceInfoView tsi={measure.data.trip} format="Short" />
              {allCanceled
                ? ": Komplettausfall"
                : `: Teilausfall (${canceled} ${
                    canceled == 1 ? "Halt" : "Halte"
                  })`}
            </>
          ) : (
            <span className="text-db-red-500">Kein Trip gewählt</span>
          )}
        </div>
      );
    }
    case "UpdateCapacitiesMeasure": {
      return (
        <div className="text-sm text-gray-500">
          {measure.data.trip ? (
            <>
              <TripServiceInfoView tsi={measure.data.trip} format="Short" />
              {`: ${measure.data.seats}`}
            </>
          ) : (
            <span className="text-db-red-500">Kein Trip gewählt</span>
          )}
        </div>
      );
    }
    case "Empty": {
      return <></>;
    }
  }
}

interface TripWithLoadLevelProps {
  tsi: TripServiceInfo | undefined;
  level: LoadLevel;
  placeholder?: string | undefined;
}

function TripWithLoadLevel({
  tsi,
  level,
  placeholder,
}: TripWithLoadLevelProps) {
  if (!tsi) {
    if (placeholder) {
      return <div className="text-db-red-500">{placeholder}</div>;
    } else {
      return null;
    }
  }
  const lli = loadLevelInfos[level];
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block w-4 h-4 rounded-full ${lli.bgColor}`} />
      <span>
        <TripServiceInfoView tsi={tsi} format="Short" />
        <span className="text-gray-500">: {lli.label}</span>
      </span>
    </div>
  );
}

interface MeasureListEntryProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  remove: RemoveFn;
  select: SelectFn;
}

function MeasureListEntry({
  measureAtom,
  remove,
  select,
}: MeasureListEntryProps) {
  const [measure] = useAtom(measureAtom);

  const hasRecipients =
    measure.shared.recipients.stations.length > 0 ||
    measure.shared.recipients.trips.length > 0;

  const needsRecipients = measureNeedsRecipients(measure);

  const tripName = (tsi: TripServiceInfo) =>
    tsi.service_infos.length > 0
      ? `${tsi.service_infos[0].category} ${tsi.service_infos[0].train_nr}`
      : `Zug ${tsi.trip.train_nr}`;

  return (
    <div className="bg-db-cool-gray-100 rounded p-3">
      <div className="flex justify-between items-center">
        <div className="font-medium text-gray-900">
          {measureTypeTexts[measure.type]}
        </div>
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-900">
            {formatDateTime(measure.shared.time)}
          </div>
          <div className="text-sm font-medium">
            <button onClick={() => select(measureAtom)} className="p-1">
              <PencilIcon className="h-5 w-5 text-gray-900 hover:text-gray-600" />
              <span className="sr-only">Bearbeiten</span>
            </button>
            <button onClick={() => remove(measureAtom)} className="p-1 ml-1">
              <TrashIcon className="h-5 w-5 text-gray-900 hover:text-gray-600" />
              <span className="sr-only">Löschen</span>
            </button>
          </div>
        </div>
      </div>
      <div className="text-sm mb-2">
        {hasRecipients ? (
          <span className="text-gray-600">
            {`Ansage in ${[
              ...measure.shared.recipients.stations.map((s) => s.name),
              ...measure.shared.recipients.trips.map((t) => tripName(t)),
            ].join(", ")}`}
          </span>
        ) : needsRecipients ? (
          <span className="text-db-red-500">Kein Ansageort gewählt</span>
        ) : null}
      </div>
      <MeasureTypeDetail measure={measure} />
    </div>
  );
}

export interface MeasureListProps {
  onSimulationFinished: () => void;
}

function MeasureList({ onSimulationFinished }: MeasureListProps): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [measureAtoms, setMeasureAtoms] = useAtom(measuresAtom);
  const [selectedMeasure, setSelectedMeasure] = useAtom(
    currentEditorMeasureAtom,
  );
  const setSimResults = useSetAtom(simResultsAtom);
  const setSelectedSimResult = useSetAtom(selectedSimResultAtom);

  const applyMeasuresMutation = useMutation(
    (measures: MeasureWrapper[]) =>
      sendPaxForecastApplyMeasuresRequest({
        universe,
        measures,
        replace_existing: true,
        preparation_time: 0,
        include_before_trip_load_info: true,
        include_after_trip_load_info: true,
        include_trips_with_unchanged_load: false,
      }),
    {
      onMutate: () => {
        return { startedAt: new Date() };
      },
      onSuccess: async (data, variables, context) => {
        console.log("measures applied");
        const result: SimulationResult = {
          universe,
          startedAt: context?.startedAt ?? new Date(),
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
        await queryClient.invalidateQueries(["tripList"]);
        onSimulationFinished();
      },
      retry: false,
    },
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
      [applyMeasuresMutation],
    ),
  );

  const add = () => {
    const systemTime = queryClient.getQueryData<PaxMonStatusResponse>(
      queryKeys.status(universe),
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
    <div
      className={cn(
        "flex flex-col gap-2 h-full overflow-hidden",
        applyMeasuresMutation.isLoading && "cursor-wait",
      )}
    >
      <div className="flex justify-between">
        <span className="text-xl">
          {`${measureAtoms.length} ${
            measureAtoms.length === 1 ? "Maßnahme" : "Maßnahmen"
          }`}
        </span>
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
          <div className="flex flex-col gap-2">
            {measureAtoms.map((ma, idx) => (
              <MeasureListEntry
                key={idx}
                measureAtom={ma}
                remove={remove}
                select={setSelectedMeasure}
              />
            ))}
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
