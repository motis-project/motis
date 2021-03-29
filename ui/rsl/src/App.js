import React, { useState, useEffect } from "react";

import { sendPaxMonStatusRequest } from "./motis/paxMonStatus";
import { sendRISForwardTimeRequest } from "./motis/risForwardTime";
import { sendPaxMonTripLoadInfoRequest } from "./motis/paxMonTripLoadInfo";
import { sendPaxMonInitForward } from "./motis/paxMonInitForward";
import { addEdgeStatistics } from "./util/statistics";

import TimeControl from "./TimeControl";
import TripPicker from "./TripPicker";
import TripLoadForecastChart from "./TripLoadForecastChart";

async function getInitialStatus(setPaxMonStatus) {
  const res = await sendPaxMonStatusRequest();
  const data = await res.json();
  setPaxMonStatus(data.content);
  if (!data.content.system_time) {
    console.log("Initial forward...");
    await sendPaxMonInitForward();
    await getInitialStatus(setPaxMonStatus);
  }
}

async function loadAndProcessTripInfo(trip) {
  const res = await sendPaxMonTripLoadInfoRequest(trip);
  const data = await res.json();
  const tli = data.content;
  addEdgeStatistics(tli);
  return tli;
}

async function forwardTimeStepped(
  endTime,
  currentTime,
  stepSize,
  setPaxMonStatus,
  selectedTrip,
  setTripLoadInfo
) {
  while (currentTime < endTime) {
    currentTime = Math.min(endTime, currentTime + stepSize);
    await sendRISForwardTimeRequest(currentTime);
    const statusRes = await sendPaxMonStatusRequest();
    const statusData = await statusRes.json();
    setPaxMonStatus(statusData.content);
    if (selectedTrip) {
      const tli = await loadAndProcessTripInfo(selectedTrip);
      setTripLoadInfo(tli);
    }
  }
}

function App() {
  const [paxMonStatus, setPaxMonStatus] = useState(null);
  const [selectedTrip, setSelectedTrip] = useState(null);
  const [tripLoadInfo, setTripLoadInfo] = useState(null);
  const [forwardInProgress, setForwardInProgress] = useState(false);

  const systemTime = paxMonStatus?.system_time;

  function forwardTime(newTime) {
    if (forwardInProgress) {
      return;
    }
    setForwardInProgress(true);
    forwardTimeStepped(
      newTime,
      systemTime,
      60,
      setPaxMonStatus,
      selectedTrip,
      setTripLoadInfo
    )
      .then(() => {
        setForwardInProgress(false);
      })
      .catch((e) => {
        console.log("forwardTime failed:", e);
        setForwardInProgress(false);
      });
  }

  function loadTripInfo(trip) {
    setSelectedTrip(trip);
    loadAndProcessTripInfo(trip).then((tli) => {
      setTripLoadInfo(tli);
    });
  }

  useEffect(() => {
    getInitialStatus(setPaxMonStatus).catch((e) => {
      console.log("getInitialStatus failed:", e);
    });
  }, []);

  return (
    <div className="App">
      <TimeControl
        systemTime={systemTime}
        onForwardTime={forwardTime}
        disabled={forwardInProgress}
      />
      <TripPicker onLoadTripInfo={loadTripInfo} />
      <TripLoadForecastChart data={tripLoadInfo} systemTime={systemTime} />
    </div>
  );
}

export default App;
