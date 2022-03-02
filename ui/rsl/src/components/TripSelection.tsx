import { useAtom } from "jotai";
import { useState } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";

import { selectedTripAtom } from "@/data/selectedTrip";

import TripList from "@/components/TripList";
import TripPicker from "@/components/TripPicker";
import TripServiceInfoView from "@/components/TripServiceInfoView";

function TripSelection(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useAtom(selectedTripAtom);
  const [trips, setTrips] = useState<TripServiceInfo[]>([]);

  function addTrip(tsi: TripServiceInfo | undefined) {
    if (tsi) {
      setTrips((ts) => [tsi, ...ts]);
    }
    setSelectedTrip(tsi?.trip);
  }

  return (
    <div>
      <div className="flex items-center justify-center gap-2">
        <span>Zugnummer:</span>
        <TripPicker
          onTripPicked={addTrip}
          clearOnPick={true}
          longDistanceOnly={false}
          className="w-64"
        />
      </div>
      <div className="flex flex-col gap-4">
        {trips.map((trip) => (
          <div
            key={JSON.stringify(trip.trip)}
            className="cursor-pointer"
            onClick={() => setSelectedTrip(trip.trip)}
          >
            <TripServiceInfoView tsi={trip} format="Long" />
          </div>
        ))}
      </div>
      <div className="mt-10">
        <TripList />
      </div>
    </div>
  );
}

export default TripSelection;
