import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai/optics";
import { useMemo } from "react";

import { Station, TripServiceInfo } from "../../api/protocol/motis";

import {
  MeasureUnion,
  isTripRecommendationMeasureU,
} from "../../data/measures";

import StationPicker from "../StationPicker";
import TripPicker from "../TripPicker";

export type TripRecommendationMeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
};

const labelClass = "font-semibold";

function TripRecommendationMeasureEditor({
  measureAtom,
}: TripRecommendationMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(
    () =>
      focusAtom(measureAtom, (optic) =>
        optic.guard(isTripRecommendationMeasureU).prop("data")
      ),
    [measureAtom]
  );
  const [data, setData] = useAtom(dataAtom);

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const setTripRecDestination = (station: Station | undefined) =>
    setData((d) => {
      return { ...d, planned_destination: station };
    });

  const setTripRecInterchange = (station: Station | undefined) =>
    setData((d) => {
      return { ...d, interchange_station: station };
    });

  const setTripRecTrip = (tsi: TripServiceInfo | undefined) =>
    setData((d) => {
      return { ...d, recommended_trip: tsi };
    });

  return (
    <div>
      {/*<div className="font-semibold mt-2">Alternativenempfehlung</div>*/}
      <div>
        <div className={labelClass}>Reisende Richtung</div>
        <StationPicker
          onStationPicked={setTripRecDestination}
          clearOnPick={false}
          initialStation={data.planned_destination}
        />
      </div>
      <div>
        <div className={labelClass}>Umsteigen an Station</div>
        <StationPicker
          onStationPicked={setTripRecInterchange}
          clearOnPick={false}
          initialStation={data.interchange_station}
        />
      </div>
      <div>
        <div className={labelClass}>in Trip</div>
        <TripPicker
          onTripPicked={setTripRecTrip}
          clearOnPick={false}
          longDistanceOnly={false}
          initialTrip={data.recommended_trip}
        />
      </div>
    </div>
  );
}

export default TripRecommendationMeasureEditor;
