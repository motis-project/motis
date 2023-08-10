import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import { useCallback, useMemo } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel } from "@/api/protocol/motis/paxforecast";

import { MeasureUnion, isTripLoadInfoMeasureU } from "@/data/measures";

import { knownLoadLevels } from "@/components/inputs/LoadInput";
import TripAndLoadInput from "@/components/sim/measures/TripAndLoadInput";

export interface TripLoadInfoMeasureEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
}

const labelClass = "font-semibold";

function TripLoadInfoMeasureEditor({
  measureAtom,
  closeEditor,
}: TripLoadInfoMeasureEditorProps): JSX.Element {
  console.log("TripLoadInfoMeasureEditor()");
  const dataAtom = useMemo(() => {
    console.log("TripLoadInfoMeasureEditor: creating dataAtom");
    return focusAtom(measureAtom, (optic) =>
      optic.guard(isTripLoadInfoMeasureU).prop("data"),
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
    [setData],
  );

  const setLoadInfoLevel = useCallback(
    (level: LoadLevel) =>
      setData((d) => {
        return { ...d, level };
      }),
    [setData],
  );

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className={labelClass}>Auslastung für Zug</div>
        <TripAndLoadInput
          selectedTrip={data.trip}
          selectedLevel={data.level}
          onTripSelected={setLoadInfoTrip}
          onLevelSelected={setLoadInfoLevel}
          loadLevels={knownLoadLevels}
        />
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
