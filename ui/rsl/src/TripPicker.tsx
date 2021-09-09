import React, { useRef, useState } from "react";

import { PaxMonTripInfo } from "./api/protocol/motis/paxmon";
import { TripId } from "./api/protocol/motis";
import { ServiceClass } from "./api/constants";
import { usePaxMonFindTripsQuery } from "./api/paxmon";
import TripView from "./TripView";

function filterTrips(trips: PaxMonTripInfo[]) {
  return trips.filter((trip) =>
    trip.tsi.service_infos.some(
      (si) =>
        si.clasz === ServiceClass.ICE ||
        si.clasz === ServiceClass.IC ||
        si.clasz === ServiceClass.OTHER
    )
  );
}

type TripPickerProps = {
  onTripPicked: (trip: TripId) => void;
};

function TripPicker({ onTripPicked }: TripPickerProps): JSX.Element {
  const trainNrInput = useRef<HTMLInputElement | null>(null);
  const [trainNr, setTrainNr] = useState<number>();
  const { data } = usePaxMonFindTripsQuery(trainNr);

  const tripList = filterTrips(data?.trips || []);

  function findByTrainNr(e: React.FormEvent) {
    e.preventDefault();
    setTrainNr(parseInt(trainNrInput.current?.value || ""));
  }

  const filterForm = (
    <div className="flex items-center m-2">
      <form className="space-x-2" onSubmit={findByTrainNr}>
        <label>
          Train number:
          <input
            type="text"
            pattern="\d+"
            ref={trainNrInput}
            className="w-20 border border-gray-200 rounded ml-2"
          />
        </label>
        <button
          type="submit"
          className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
        >
          Find
        </button>
      </form>
    </div>
  );

  const resultList = (
    <div className="m-2">
      <ul>
        {tripList.map((data, idx) => (
          <li
            key={idx.toString()}
            onClick={() => onTripPicked(data.tsi.trip)}
            className="cursor-pointer hover:underline"
          >
            <TripView tsi={data.tsi} format="Long" />
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <div>
      {filterForm}
      {resultList}
    </div>
  );
}

export default TripPicker;
