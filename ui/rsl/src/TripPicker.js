import { useState } from "react";

import { sendPaxMonFindTripsRequest } from "./motis/paxMonFindTrips";
import { formatDateTime } from "./util/dateFormat";

function TripView(props) {
  const names = [
    ...new Set(
      props.data.tsi.service_infos.map((si) =>
        si.line ? `${si.name} [${si.train_nr}]` : si.name
      )
    ),
  ];
  return (
    <span>
      {names.join(", ")} ({props.data.tsi.primary_station.name} (
      {formatDateTime(props.data.tsi.trip.time)}) –{" "}
      {props.data.tsi.secondary_station.name} (
      {formatDateTime(props.data.tsi.trip.target_time)}))
    </span>
  );
}

function filterTrips(trips) {
  return trips.filter((trip) =>
    trip.tsi.service_infos.some(
      (si) => si.clasz === 1 || si.clasz === 2 || si.clasz === 12
    )
  );
}

function TripPicker(props) {
  const [trainNrText, setTrainNrText] = useState("");
  const [tripList, setTripList] = useState([]);

  function findByTrainNr(e) {
    e.preventDefault();
    const trainNr = parseInt(trainNrText);
    if (trainNr) {
      sendPaxMonFindTripsRequest({ train_nr: trainNr })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);
          setTripList(filterTrips(data.content.trips));
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
            onClick={() => props.onLoadTripInfo(data.tsi.trip)}
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
