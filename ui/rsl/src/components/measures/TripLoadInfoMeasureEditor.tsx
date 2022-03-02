import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai/optics";
import { useCallback, useMemo } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel } from "@/api/protocol/motis/paxforecast";

import { MeasureUnion, isTripLoadInfoMeasureU } from "@/data/measures";

import TripPicker from "@/components/TripPicker";

export type TripLoadInfoMeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
};

const labelClass = "font-semibold";

const loadLevels: Array<{ level: LoadLevel; label: string }> = [
  { level: "Unknown", label: "unbekannt" },
  { level: "Low", label: "gering" },
  { level: "NoSeats", label: "keine Sitzplätze mehr" },
  { level: "Full", label: "keine Mitfahrmöglichkeit mehr" },
];

function TripLoadInfoMeasureEditor({
  measureAtom,
  closeEditor,
}: TripLoadInfoMeasureEditorProps): JSX.Element {
  console.log("TripLoadInfoMeasureEditor()");
  const dataAtom = useMemo(() => {
    console.log("TripLoadInfoMeasureEditor: creating dataAtom");
    return focusAtom(measureAtom, (optic) =>
      optic.guard(isTripLoadInfoMeasureU).prop("data")
    );
  }, [measureAtom]);
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setLoadInfoTrip = useCallback(
    (tsi: TripServiceInfo | undefined) =>
      setData((d) => {
        return { ...d, trip: tsi };
      }),
    [setData]
  );

  const setLoadInfoLevel = useCallback(
    (level: LoadLevel) =>
      setData((d) => {
        return { ...d, level };
      }),
    [setData]
  );

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className={labelClass}>Trip</div>
        <div>
          <TripPicker
            onTripPicked={setLoadInfoTrip}
            clearOnPick={false}
            longDistanceOnly={false}
            initialTrip={data.trip}
          />
        </div>
      </div>
      <div>
        <div className={labelClass}>Auslastungsstufe</div>
        <div className="flex flex-col">
          {loadLevels.map(({ level, label }) => (
            <label key={level} className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="load-level"
                value={level}
                checked={data.level == level}
                onChange={() => setLoadInfoLevel(level)}
              />
              {label}
            </label>
          ))}
        </div>
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

export default TripLoadInfoMeasureEditor;
