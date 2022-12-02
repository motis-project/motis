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

import classNames from "@/util/classNames";

import { GroupDetailsFromRoute } from "@/components/groups/GroupDetails";
import GroupsMainSection from "@/components/groups/GroupsMainSection";
import Header from "@/components/header/Header";
import IndexPage from "@/components/index/IndexPage";
import SimPanel from "@/components/sim/SimPanel";
import GroupStatistics from "@/components/stats/GroupStatistics";
import { TripDetailsFromRoute } from "@/components/trips/TripDetails";
import TripsMainSection from "@/components/trips/TripsMainSection";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: true, staleTime: 10000 },
  },
});

function MainContent(): JSX.Element {
  const [showSimPanel] = useAtom(showSimPanelAtom);
  return (
    <div className="flex justify-between items-stretch overflow-y-auto grow">
      <Outlet />
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

function Root(): JSX.Element {
  return (
    <div className="w-full h-screen flex flex-col">
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
        path: "groups",
        element: <GroupsMainSection />,
        children: [{ path: ":groupId", element: <GroupDetailsFromRoute /> }],
      },
      { path: "stats", element: <GroupStatistics /> },
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
