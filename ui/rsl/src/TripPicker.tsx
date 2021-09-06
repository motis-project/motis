import React, { useRef, useState } from "react";
import { useQuery } from "react-query";

import { PaxMonTripInfo } from "./api/protocol/motis/paxmon";
import { TripId } from "./api/protocol/motis";
import { sendPaxMonFindTripsRequest } from "./api/paxmon";
import TripView from "./TripView";

function filterTrips(trips: PaxMonTripInfo[]) {
  return trips.filter((trip) =>
    trip.tsi.service_infos.some(
      (si) => si.clasz === 1 || si.clasz === 2 || si.clasz === 12
    )
  );
}

type TripPickerProps = {
  onTripPicked: (trip: TripId) => void;
};

function TripPicker({ onTripPicked }: TripPickerProps): JSX.Element {
  const trainNrInput = useRef<HTMLInputElement | null>(null);
  const [trainNr, setTrainNr] = useState<number | null>(null);
  const { data: tripList } = useQuery(
    ["trips", trainNr],
    async () => {
      const res = await sendPaxMonFindTripsRequest({
        universe: 0,
        train_nr: trainNr || 0,
        only_trips_with_paxmon_data: true,
        filter_class: false,
        max_class: 0,
      });
      return filterTrips(res.trips);
    },
    { enabled: trainNr !== null }
  );

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

  const resultList = tripList && (
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
