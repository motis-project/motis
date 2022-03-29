import { TrashIcon } from "@heroicons/react/solid";
import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai/optics";
import React, { ReactNode, useCallback, useMemo } from "react";

import { Station, TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel } from "@/api/protocol/motis/paxforecast";

import {
  MeasureUnion,
  isTripLoadRecommendationMeasureU,
} from "@/data/measures";

import StationPicker from "@/components/StationPicker";
import TripPicker from "@/components/TripPicker";
import LoadInput, {
  allLoadLevels,
  highLoadLevels,
  lowLoadLevels,
} from "@/components/measures/LoadInput";

export type TripLoadRecommendationMeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
};

const labelClass = "font-semibold";

function TripLoadRecommendationMeasureEditor({
  measureAtom,
  closeEditor,
}: TripLoadRecommendationMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(() => {
    console.log("TripLoadInfoMeasureEditor: creating dataAtom");
    return focusAtom(measureAtom, (optic) =>
      optic.guard(isTripLoadRecommendationMeasureU).prop("data")
    );
  }, [measureAtom]);
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setPlannedDestination = useCallback(
    (station: Station | undefined) =>
      setData((d) => {
        return { ...d, planned_destination: station };
      }),
    [setData]
  );

  const setFullTrip = useCallback(
    (tsi: TripServiceInfo | undefined) =>
      setData((d) => {
        return { ...d, full_trip: { ...d.full_trip, trip: tsi } };
      }),
    [setData]
  );

  const setFullTripLevel = useCallback(
    (level: LoadLevel) =>
      setData((d) => {
        return { ...d, full_trip: { ...d.full_trip, level } };
      }),
    [setData]
  );

  const setAlternativeTrip = useCallback(
    (idx: number, tsi: TripServiceInfo | undefined) =>
      setData((d) => {
        const newTrips = [...d.recommended_trips];
        newTrips[idx] = { ...newTrips[idx], trip: tsi };
        return { ...d, recommended_trips: newTrips };
      }),
    [setData]
  );

  const setAlternativeTripLevel = useCallback(
    (idx: number, level: LoadLevel) =>
      setData((d) => {
        const newTrips = [...d.recommended_trips];
        newTrips[idx] = { ...newTrips[idx], level };
        return { ...d, recommended_trips: newTrips };
      }),
    [setData]
  );

  const removeAlternative = useCallback(
    (idx: number) =>
      setData((d) => {
        const newTrips = [...d.recommended_trips];
        newTrips.splice(idx, 1);
        return { ...d, recommended_trips: newTrips };
      }),
    [setData]
  );

  const addAlternative = useCallback(
    () =>
      setData((d) => {
        return {
          ...d,
          recommended_trips: [
            ...d.recommended_trips,
            { trip: undefined, level: "Low" },
          ],
        };
      }),
    [setData]
  );

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className={labelClass}>Ansage für Reisende Richtung</div>
        <StationPicker
          onStationPicked={setPlannedDestination}
          clearOnPick={false}
          initialStation={data.planned_destination}
        />
      </div>

      <div>
        <div className={labelClass}>Überlasteter Zug</div>
        <TripAndLoadInput
          selectedTrip={data.full_trip.trip}
          selectedLevel={data.full_trip.level}
          onTripSelected={setFullTrip}
          onLevelSelected={setFullTripLevel}
          loadLevels={highLoadLevels}
        />
      </div>

      <div>
        <div className={labelClass}>Alternativen:</div>
        <div className="flex flex-col gap-2">
          {data.recommended_trips.map((tll, idx) => (
            <div key={idx} className="flex flex-col">
              <TripAndLoadInput
                selectedTrip={tll.trip}
                selectedLevel={tll.level}
                onTripSelected={(tsi) => setAlternativeTrip(idx, tsi)}
                onLevelSelected={(level) => setAlternativeTripLevel(idx, level)}
                loadLevels={lowLoadLevels}
              >
                <button
                  type="button"
                  className="p-1"
                  onClick={() => removeAlternative(idx)}
                >
                  <TrashIcon className="h-4 w-4 text-gray-900 hover:text-gray-600" />
                  <span className="sr-only">Löschen</span>
                </button>
              </TripAndLoadInput>
            </div>
          ))}
          <button
            onClick={addAlternative}
            className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-sm rounded"
          >
            Weitere Alternative hinzufügen
          </button>
        </div>
      </div>

      <button
        onClick={() => closeEditor()}
        className="mt-4 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white rounded"
      >
        Maßnahme speichern
      </button>
    </div>
  );
}

type TripAndLoadInputProps = {
  selectedTrip: TripServiceInfo | undefined;
  selectedLevel: LoadLevel;
  onTripSelected: (tsi: TripServiceInfo | undefined) => void;
  onLevelSelected: (level: LoadLevel) => void;
  loadLevels?: LoadLevel[];
  children?: ReactNode;
};

function TripAndLoadInput({
  selectedTrip,
  selectedLevel,
  onTripSelected,
  onLevelSelected,
  loadLevels = allLoadLevels,
  children,
}: TripAndLoadInputProps) {
  return (
    <div className="flex justify-between items-center gap-2">
      <TripPicker
        onTripPicked={onTripSelected}
        clearOnPick={false}
        longDistanceOnly={false}
        initialTrip={selectedTrip}
        key={JSON.stringify(selectedTrip)}
        className="w-32 flex-shrink-0"
      />
      <LoadInput
        loadLevels={loadLevels}
        selectedLevel={selectedLevel}
        onLevelSelected={onLevelSelected}
        className="flex-grow"
      />
      {children}
    </div>
  );
}

export default TripLoadRecommendationMeasureEditor;
