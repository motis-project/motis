import { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import { TripId } from "@/api/protocol/motis";

import getQueryParameters from "@/util/queryParameters";

import SimPanel from "@/components/SimPanel";
import TimeControl from "@/components/TimeControl";
import TripDetails from "@/components/TripDetails";
import TripSelection from "@/components/TripSelection";
import UniverseControl from "@/components/UniverseControl";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

const allowForwarding = getQueryParameters()["allowForwarding"] === "yes";

function App(): JSX.Element {
  const [selectedTrip, setSelectedTrip] = useState<TripId>();

  return (
    <QueryClientProvider client={queryClient}>
      <div className="w-full h-screen flex flex-col">
        <div
          className="flex justify-center items-baseline space-x-4 p-2
            bg-db-cool-gray-200 text-black divide-x-2 divide-db-cool-gray-400"
        >
          <TimeControl allowForwarding={allowForwarding} />
          <UniverseControl />
        </div>

        <div className="flex justify-between items-stretch overflow-y-auto grow">
          <div className="bg-db-cool-gray-200 w-[24rem] overflow-y-auto p-2">
            <TripSelection onTripSelected={setSelectedTrip} />
          </div>
          <div className="overflow-y-auto grow p-2">
            {selectedTrip !== undefined ? (
              <TripDetails tripId={selectedTrip} />
            ) : null}
          </div>
          <div className="bg-db-cool-gray-200 overflow-y-auto grow max-w-xl p-2">
            <SimPanel />
          </div>
        </div>
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
