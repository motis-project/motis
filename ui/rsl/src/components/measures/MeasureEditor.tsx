import { PrimitiveAtom, useAtom } from "jotai";
import { useUpdateAtom } from "jotai/utils";

import { MeasureUnion } from "../../data/measures";

import RtUpdateMeasureEditor from "./RtUpdateMeasureEditor";
import SharedDataEditor from "./SharedDataEditor";
import TripLoadInfoMeasureEditor from "./TripLoadInfoMeasureEditor";
import TripRecommendationMeasureEditor from "./TripRecommendationMeasureEditor";

export type MeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
};

function MeasureEditor({ measureAtom }: MeasureEditorProps): JSX.Element {
  const [measure, setMeasure] = useAtom(measureAtom);

  const measureEditor = (e: JSX.Element) => (
    <div>
      Editor for {`${measureAtom}`}:
      <div>
        <button
          onClick={() =>
            setMeasure((m) => {
              return { type: "Empty", shared: m.shared };
            })
          }
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
        >
          Maßnahmentyp ändern
        </button>
      </div>
      <SharedDataEditor measureAtom={measureAtom} />
      {e}
    </div>
  );

  switch (measure.type) {
    case "Empty":
      return (
        <EmptyMeasureEditor
          measureAtom={measureAtom}
          key={measureAtom.toString()}
        />
      );
    case "TripLoadInfoMeasure":
      return measureEditor(
        <TripLoadInfoMeasureEditor
          measureAtom={measureAtom}
          key={measureAtom.toString()}
        />
      );
    case "TripRecommendationMeasure":
      return measureEditor(
        <TripRecommendationMeasureEditor
          measureAtom={measureAtom}
          key={measureAtom.toString()}
        />
      );
    case "RtUpdateMeasure":
      return measureEditor(
        <RtUpdateMeasureEditor
          measureAtom={measureAtom}
          key={measureAtom.toString()}
        />
      );
  }
}

function EmptyMeasureEditor({ measureAtom }: MeasureEditorProps) {
  const setMeasure = useUpdateAtom(measureAtom);

  const setTripLoadInfo = () => {
    setMeasure((m) => {
      return {
        type: "TripLoadInfoMeasure",
        shared: m.shared,
        data: { trip: undefined, level: "Unknown" },
      };
    });
  };

  const setTripRecommendation = () => {
    setMeasure((m) => {
      return {
        type: "TripRecommendationMeasure",
        shared: m.shared,
        data: {
          planned_destination: undefined,
          recommended_trip: undefined,
          interchange_station: undefined,
        },
      };
    });
  };

  const setRtUpdate = () => {
    setMeasure((m) => {
      return {
        type: "RtUpdateMeasure",
        shared: m.shared,
        data: { trip: undefined, ribasis: undefined },
      };
    });
  };

  return (
    <div>
      Maßnahmentyp wählen:
      <div>
        <button
          onClick={setTripLoadInfo}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
        >
          Auslastungsinformation
        </button>
        <button
          onClick={setTripRecommendation}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
        >
          Alternativenempfehlung
        </button>
        <button
          onClick={setRtUpdate}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
        >
          Echtzeitupdate
        </button>
      </div>
    </div>
  );
}

export default MeasureEditor;
