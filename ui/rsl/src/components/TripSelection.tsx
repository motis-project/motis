import { useUpdateAtom } from "jotai/utils";

import { selectedTripAtom } from "@/data/selectedTrip";

import TripList from "@/components/TripList";
import TripPicker from "@/components/TripPicker";

function TripSelection(): JSX.Element {
  const setSelectedTrip = useUpdateAtom(selectedTripAtom);

  return (
    <div className="h-full flex flex-col ">
      <div className="flex items-center justify-center gap-2">
        <span>Zug suchen:</span>
        <TripPicker
          onTripPicked={setSelectedTrip}
          clearOnPick={true}
          longDistanceOnly={false}
          className="w-64"
        />
      </div>
      <div className="mt-5 grow">
        <TripList />
      </div>
    </div>
  );
}

export default TripSelection;
