import React, { useState, useEffect } from "react";

import { sendPaxMonStatusRequest } from "./motis/paxMonStatus";
import { sendRISForwardTimeRequest } from "./motis/risForwardTime";
import { sendPaxMonTripLoadInfoRequest } from "./motis/paxMonTripLoadInfo";
import { sendPaxMonInitForward } from "./motis/paxMonInitForward";
import { addEdgeStatistics } from "./util/statistics";

import TimeControl from "./TimeControl";
import TripPicker from "./TripPicker";
import TripLoadForecastChart from "./TripLoadForecastChart";

function updateStatus(setPaxMonStatus) {
  sendPaxMonStatusRequest({ include_trips_affected_by_last_update: true })
    .then((res) => res.json())
    .then((data) => {
      console.log(data.content);
      setPaxMonStatus(data.content);
      if (!data.content.system_time) {
        console.log("Initial forward...");
        sendPaxMonInitForward().then((res) => updateStatus(setPaxMonStatus));
      }
    });
}

function App() {
  const [paxMonStatus, setPaxMonStatus] = useState(null);
  const [tripLoadInfo, setTripLoadInfo] = useState(null);

  const systemTime = paxMonStatus?.system_time;

  function forwardTimeStepped(endTime, currentTime, stepSize) {
    const newTime = Math.min(endTime, currentTime + stepSize);
    if (newTime > currentTime) {
      sendRISForwardTimeRequest(newTime).then(() => {
        forwardTimeStepped(endTime, newTime, stepSize);
      });
    } else {
      updateStatus(setPaxMonStatus);
    }
  }

  function forwardTime(newTime) {
    forwardTimeStepped(newTime, systemTime, 60);
  }

  function loadTripInfo(trip) {
    console.log("loadTripInfo:", trip);
    sendPaxMonTripLoadInfoRequest(trip)
      .then((res) => res.json())
      .then((data) => {
        const tli = data.content;
        addEdgeStatistics(tli);
        setTripLoadInfo(tli);
        console.log(tli);
      });
  }

  useEffect(() => {
    updateStatus(setPaxMonStatus);
  }, []);

  return (
    <div className="App">
      <TimeControl systemTime={systemTime} onForwardTime={forwardTime} />
      <TripPicker onLoadTripInfo={loadTripInfo} />
      <TripLoadForecastChart data={tripLoadInfo} systemTime={systemTime} />
    </div>
  );
}

export default App;
