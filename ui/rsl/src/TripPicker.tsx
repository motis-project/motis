import React, { useState } from "react";

import { sendPaxMonFindTripsRequest } from "./motis/paxMonFindTrips";
import { formatDateTime } from "./util/dateFormat";
import { TripId } from "./motis/base";
import { PaxMonFindTripsResponse, PaxMonTripInfo } from "./motis/paxmon";

type TripViewProps = {
  data: PaxMonTripInfo;
};

function TripView({ data }: TripViewProps) {
  const names = [
    ...new Set(
      data.tsi.service_infos.map((si) =>
        si.line ? `${si.name} [${si.train_nr}]` : si.name
      )
    ),
  ];
  return (
    <span>
      {names.join(", ")} ({data.tsi.primary_station.name} (
      {formatDateTime(data.tsi.trip.time)}) – {data.tsi.secondary_station.name}{" "}
      ({formatDateTime(data.tsi.trip.target_time)}))
    </span>
  );
}

function filterTrips(trips: PaxMonTripInfo[]) {
  return trips.filter((trip) =>
    trip.tsi.service_infos.some(
      (si) => si.clasz === 1 || si.clasz === 2 || si.clasz === 12
    )
  );
}

type TripPickerProps = {
  onLoadTripInfo: (trip: TripId) => void;
};

function TripPicker({ onLoadTripInfo }: TripPickerProps): JSX.Element {
  const [trainNrText, setTrainNrText] = useState("");
  const [tripList, setTripList] = useState<PaxMonTripInfo[]>([]);

  function findByTrainNr(e: React.MouseEvent) {
    e.preventDefault();
    const trainNr = parseInt(trainNrText);
    if (trainNr) {
      sendPaxMonFindTripsRequest({ train_nr: trainNr })
        .then((res) => res.json())
        .then((msg) => {
          console.log(msg);
          const data = msg.content as PaxMonFindTripsResponse;
          setTripList(filterTrips(data.trips));
        });
    }
  }

  const filterForm = (
    <div className="flex items-center m-2">
      <form className="space-x-2">
        <label>
          Train number:
          <input
            type="text"
            pattern="\d+"
            value={trainNrText}
            onChange={(e) => setTrainNrText(e.target.value)}
            className="w-20 border border-gray-200 rounded ml-2"
          />
        </label>
        <button
          type="submit"
          className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
          onClick={findByTrainNr}
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
            onClick={() => onLoadTripInfo(data.tsi.trip)}
            className="cursor-pointer hover:underline"
          >
            <TripView data={data} />
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
