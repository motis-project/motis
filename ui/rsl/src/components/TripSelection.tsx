import { useState } from "react";

import { TripId, TripServiceInfo } from "../api/protocol/motis";

import CriticalTripList from "./CriticialTripList";
import TripPicker from "./TripPicker";
import TripServiceInfoView from "./TripServiceInfoView";

export type TripSelectionProps = {
  onTripSelected: (trip: TripId | undefined) => void;
};

function TripSelection({ onTripSelected }: TripSelectionProps): JSX.Element {
  const [trips, setTrips] = useState<TripServiceInfo[]>([]);

  function addTrip(tsi: TripServiceInfo | undefined) {
    if (tsi) {
      setTrips((ts) => [tsi, ...ts]);
    }
    onTripSelected(tsi?.trip);
  }

  return (
    <div>
      <div className="flex items-center justify-center gap-2">
        <span>Zugnummer:</span>
        <TripPicker
          onTripPicked={addTrip}
          clearOnPick={true}
          longDistanceOnly={false}
          className="w-96"
        />
      </div>
      <div className="flex flex-col gap-4">
        {trips.map((trip) => (
          <div
            key={JSON.stringify(trip.trip)}
            className="cursor-pointer"
            onClick={() => onTripSelected(trip.trip)}
          >
            <TripServiceInfoView tsi={trip} format="Long" />
          </div>
        ))}
      </div>
      <div className="mt-10">
        <CriticalTripList onTripSelected={onTripSelected} />
      </div>
    </div>
  );
}

export default TripSelection;
