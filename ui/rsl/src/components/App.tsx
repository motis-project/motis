import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { useAtom } from "jotai";

import { mainPageAtom, showSimPanelAtom } from "@/data/views";

import classNames from "@/util/classNames";

import GroupsMainSection from "@/components/groups/GroupsMainSection";
import Header from "@/components/header/Header";
import SimPanel from "@/components/sim/SimPanel";
import GroupStatistics from "@/components/stats/GroupStatistics";
import TripsMainSection from "@/components/trips/TripsMainSection";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

function MainPage(): JSX.Element {
  const [mainPage] = useAtom(mainPageAtom);

  return (
    <>
      <TripsMainSection visible={mainPage === "trips"} />
      <GroupsMainSection visible={mainPage === "groups"} />
      <GroupStatistics visible={mainPage === "stats"} />
    </>
  );
}

function MainContent(): JSX.Element {
  const [showSimPanel] = useAtom(showSimPanelAtom);
  return (
    <div className="flex justify-between items-stretch overflow-y-auto grow">
      <MainPage />
      <div
        className={classNames(
          "bg-db-cool-gray-200 dark:bg-gray-800 overflow-y-auto p-2 w-[32rem] shrink-0",
          showSimPanel ? "block" : "hidden"
        )}
      >
        <SimPanel />
      </div>
    </div>
  );
}

function App(): JSX.Element {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="w-full h-screen flex flex-col">
        <Header />
        <MainContent />
      </div>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
