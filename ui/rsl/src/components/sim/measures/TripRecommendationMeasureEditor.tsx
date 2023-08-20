import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import { useMemo } from "react";

import { Station, TripServiceInfo } from "@/api/protocol/motis";

import { MeasureUnion, isTripRecommendationMeasureU } from "@/data/measures";

import StationPicker from "@/components/inputs/StationPicker";
import TripPicker from "@/components/inputs/TripPicker";

export interface TripRecommendationMeasureEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
}

const labelClass = "font-semibold";

function TripRecommendationMeasureEditor({
  measureAtom,
  closeEditor,
}: TripRecommendationMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(
    () =>
      focusAtom(measureAtom, (optic) =>
        optic.guard(isTripRecommendationMeasureU).prop("data"),
      ),
    [measureAtom],
  );
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setTripRecDestination = (station: Station | undefined) =>
    setData((d) => {
      return { ...d, planned_destination: station };
    });

  const setTripRecTrip = (tsi: TripServiceInfo | undefined) =>
    setData((d) => {
      return { ...d, recommended_trip: tsi };
    });

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className={labelClass}>Reisende Richtung</div>
        <StationPicker
          onStationPicked={setTripRecDestination}
          clearOnPick={false}
          clearButton={false}
          initialStation={data.planned_destination}
        />
      </div>
      <div>
        <div className={labelClass}>Umsteigen in Zug</div>
        <TripPicker
          onTripPicked={setTripRecTrip}
          clearOnPick={false}
          longDistanceOnly={false}
          initialTrip={data.recommended_trip}
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

export default TripRecommendationMeasureEditor;
