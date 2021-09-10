import React, { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import TimeControl from "./TimeControl";
import TripPicker from "./TripPicker";
import TripLoadForecastChart from "./TripLoadForecastChart";
import { TripId } from "./api/protocol/motis";
import TripSectionDetails from "./TripSectionDetails";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, staleTime: 10000 },
  },
});

function App(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useState<TripId | null>(null);

  const tripDisplay =
    selectedTrip !== null ? (
      <>
        <TripLoadForecastChart tripId={selectedTrip} />
        <TripSectionDetails tripId={selectedTrip} />
      </>
    ) : null;

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
