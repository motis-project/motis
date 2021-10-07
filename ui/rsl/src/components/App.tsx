import React, { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import { TripId } from "../api/protocol/motis";
import getQueryParameters from "../util/queryParameters";

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

const allowForwarding = getQueryParameters()["allowForwarding"] === "yes";

function App(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useState<TripId>();
  const [simActive, setSimActive] = useState(false);

  const tripDisplay =
    selectedTrip !== undefined ? <TripDetails tripId={selectedTrip} /> : null;

  return (
    <QueryClientProvider client={queryClient}>
      <div>
        <div
          className="fixed top-0 w-full z-20 flex justify-center items-baseline space-x-4 p-2
            bg-db-cool-gray-200 text-black divide-x-2 divide-db-cool-gray-400"
        >
          <TimeControl allowForwarding={allowForwarding} />
          <UniverseControl />
          <div className="flex pl-4">
            <button
              type="button"
              className="bg-db-red-500 px-3 py-1 rounded text-white text-sm hover:bg-db-red-600"
              onClick={() => setSimActive((active) => !active)}
            >
              Maßnahmensimulation
            </button>
          </div>
        </div>

        <div className="flex justify-between w-full mt-10">
          {simActive && (
            <div
              className="h-screen sticky top-0 pt-10 flex flex-col w-full bg-db-cool-gray-200
              w-80 p-2 shadow-md overflow-visible"
            >
              <MeasureInput />
            </div>
          )}
          <div className="flex-grow">
            <div className="mt-6 flex items-center justify-center gap-2">
              <span>Trip:</span>
              <TripPicker
                onTripPicked={(tsi) => setSelectedTrip(tsi?.trip)}
                clearOnPick={false}
                longDistanceOnly={true}
                className="w-96"
              />
            </div>
            {tripDisplay}
          </div>
        </div>
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
