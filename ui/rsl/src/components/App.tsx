import React, { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import { TripId } from "../api/protocol/motis";

import TimeControl from "./TimeControl";
import TripPicker from "./TripPicker";
import TripDetails from "./TripDetails";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

function App(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useState<TripId | null>(null);

  const tripDisplay =
    selectedTrip !== null ? <TripDetails tripId={selectedTrip} /> : null;

  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <TimeControl />
        <TripPicker onTripPicked={setSelectedTrip} />
        {tripDisplay}
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
