import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import getQueryParameters from "@/util/queryParameters";

import MainSection from "@/components/MainSection";
import Settings from "@/components/Settings";
import SimPanel from "@/components/SimPanel";
import TimeControl from "@/components/TimeControl";
import TripList from "@/components/TripList";
import UniverseControl from "@/components/UniverseControl";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

const allowForwarding = getQueryParameters()["allowForwarding"] === "yes";

function App(): JSX.Element {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="w-full h-screen flex flex-col">
        <div
          className="flex justify-center items-baseline space-x-4 p-2
            bg-db-cool-gray-200 dark:bg-gray-800 text-black dark:text-neutral-300 divide-x-2 divide-db-cool-gray-400"
        >
          <TimeControl allowForwarding={allowForwarding} />
          <UniverseControl />
        </div>
        <Settings />

        <div className="flex justify-between items-stretch overflow-y-auto grow">
          <div className="bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0">
            <TripList />
          </div>
          <div className="overflow-y-auto grow p-2">
            <MainSection />
          </div>
          <div className="bg-db-cool-gray-200 dark:bg-gray-800 overflow-y-auto p-2 w-[32rem] shrink-0">
            <SimPanel />
          </div>
        </div>
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
