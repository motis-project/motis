import React, { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import { TripId } from "../api/protocol/motis";

import TimeControl from "./TimeControl";
import TripPicker from "./TripPicker";
import TripDetails from "./TripDetails";
import MeasureInput from "./measures/MeasureInput";
import UniverseControl from "./UniverseControl";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

function App(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useState<TripId>();
  const [simActive, setSimActive] = useState(false);

  const tripDisplay =
    selectedTrip !== undefined ? <TripDetails tripId={selectedTrip} /> : null;

  return (
    <QueryClientProvider client={queryClient}>
      <div>
        <div className="flex justify-center items-baseline space-x-2 p-2 bg-blue-600 text-white divide-x-2 divide-white">
          <TimeControl allowForwarding={true} />
          <UniverseControl />
          <div className="flex pl-2">
            <button
              type="button"
              className="bg-blue-800 px-3 py-1 rounded-xl text-white text-sm hover:bg-blue-800"
              onClick={() => setSimActive((active) => !active)}
            >
              Maßnahmensimulation
            </button>
          </div>
        </div>
        {simActive && <MeasureInput />}
        <div className="mt-4 flex items-center justify-center gap-2">
          <span>Trip:</span>
          <TripPicker
            onTripPicked={(tsi) => setSelectedTrip(tsi?.trip)}
            clearOnPick={false}
            longDistanceOnly={true}
          />
        </div>
        {tripDisplay}
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
