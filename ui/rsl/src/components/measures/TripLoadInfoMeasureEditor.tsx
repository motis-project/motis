import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai/optics";
import { useMemo } from "react";

import { TripServiceInfo } from "../../api/protocol/motis";
import { LoadLevel } from "../../api/protocol/motis/paxforecast";

import { MeasureUnion, isTripLoadInfoMeasureU } from "../../data/measures";

import TripPicker from "../TripPicker";

export type TripLoadInfoMeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
};

const labelClass = "font-semibold";

const loadLevels: Array<{ level: LoadLevel; label: string }> = [
  { level: "Unknown", label: "unbekannt" },
  { level: "Low", label: "gering" },
  { level: "NoSeats", label: "keine Sitzplätze mehr" },
  { level: "Full", label: "voll" },
];

function TripLoadInfoMeasureEditor({
  measureAtom,
}: TripLoadInfoMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(
    () =>
      focusAtom(measureAtom, (optic) =>
        optic.guard(isTripLoadInfoMeasureU).prop("data")
      ),
    [measureAtom]
  );
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setLoadInfoTrip = (tsi: TripServiceInfo | undefined) =>
    setData((d) => {
      return { ...d, trip: tsi };
    });

  const setLoadInfoLevel = (level: LoadLevel) =>
    setData((d) => {
      return { ...d, level };
    });

  return (
    <div>
      {/*<div className="font-semibold mt-2">Auslastungsinformation</div>*/}
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
    </div>
  );
}

export default TripLoadInfoMeasureEditor;
