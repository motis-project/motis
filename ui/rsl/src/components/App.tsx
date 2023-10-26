import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { useAtom } from "jotai";
import {
  Outlet,
  RouterProvider,
  createHashRouter,
  redirect,
} from "react-router-dom";

import { showSimPanelAtom } from "@/data/views";

import { GroupDetailsFromRoute } from "@/components/groups/GroupDetails";
import GroupsMainSection from "@/components/groups/GroupsMainSection";
import Header from "@/components/header/Header";
import IndexPage from "@/components/index/IndexPage";
import SimPanel from "@/components/sim/SimPanel";
import GroupStatistics from "@/components/stats/GroupStatistics";
import StatusOverview from "@/components/status/StatusOverview";
import { TransferDetailsFromRoute } from "@/components/transfers/TransferDetails";
import TransfersMainSection from "@/components/transfers/TransfersMainSection";
import { TripDetailsFromRoute } from "@/components/trips/TripDetails";
import TripsMainSection from "@/components/trips/TripsMainSection";

import { cn } from "@/lib/utils";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

function MainContent(): JSX.Element {
  const [showSimPanel] = useAtom(showSimPanelAtom);
  return (
    <div className="flex grow items-stretch justify-between overflow-y-auto">
      <Outlet />
      <div
        className={cn(
          "w-[32rem] shrink-0 overflow-y-auto bg-db-cool-gray-200 p-2 dark:bg-gray-800",
          showSimPanel ? "block" : "hidden",
        )}
      >
        <SimPanel />
      </div>
    </div>
  );
}

function Root(): JSX.Element {
  return (
    <div className="flex h-screen w-full flex-col">
      <Header />
      <MainContent />
    </div>
  );
}

const router = createHashRouter([
  {
    path: "/",
    element: <Root />,
    children: [
      {
        path: "",
        element: <IndexPage />,
        loader: () => {
          return redirect("/trips");
        },
      },
      {
        path: "trips",
        element: <TripsMainSection />,
        children: [{ path: ":tripId", element: <TripDetailsFromRoute /> }],
      },
      {
        path: "transfers",
        element: <TransfersMainSection />,
        children: [{ path: ":n/:e", element: <TransferDetailsFromRoute /> }],
      },
      {
        path: "groups",
        element: <GroupsMainSection />,
        children: [{ path: ":groupId", element: <GroupDetailsFromRoute /> }],
      },
      { path: "stats", element: <GroupStatistics /> },
      { path: "status", element: <StatusOverview /> },
    ],
  },
]);

function App(): JSX.Element {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
