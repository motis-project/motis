import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import React, { ChangeEvent, useCallback, useMemo } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";

import { MeasureUnion, isUpdateCapacitiesMeasureU } from "@/data/measures";

import TripPicker from "@/components/inputs/TripPicker";

export interface UpdateCapacityMeasureEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
  deleteMeasure: (measureAtom: PrimitiveAtom<MeasureUnion>) => void;
}

const labelClass = "font-semibold";

function UpdateCapacityMeasureEditor({
  measureAtom,
  closeEditor,
}: UpdateCapacityMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(
    () =>
      focusAtom(measureAtom, (optic) =>
        optic.guard(isUpdateCapacitiesMeasureU).prop("data"),
      ),
    [measureAtom],
  );
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setTrip = useCallback(
    (tsi: TripServiceInfo | undefined) => {
      setData((d) => {
        return { ...d, trip: tsi };
      });
    },
    [setData],
  );

  const setSeats = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setData((d) => {
        return { ...d, seats: e.target.valueAsNumber };
      });
    },
    [setData],
  );

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className={labelClass}>Zug</div>
        <div>
          <TripPicker
            onTripPicked={setTrip}
            clearOnPick={false}
            longDistanceOnly={false}
            initialTrip={data.trip}
          />
        </div>
      </div>

      <div>
        <div className={labelClass}>Kapazität</div>
        <div>
          <input
            type="number"
            className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            value={data.seats}
            onChange={setSeats}
            min={0}
          />
        </div>
      </div>

      <div>
        Hinweis: Eine frühere Kapazitätsänderung kann rückgängig gemacht werden,
        indem die Kapazität oben auf 0 gesetzt wird.
      </div>

      <button
        onClick={() => closeEditor()}
        className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white rounded"
      >
        Maßnahme speichern
      </button>
    </div>
  );
}

export default UpdateCapacityMeasureEditor;
