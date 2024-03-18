import { ReactNode } from "react";
import { Outlet } from "react-router-dom";

import TripList from "@/components/trips/TripList";

function TripsMainSection(): ReactNode {
  return (
    <>
      <div className="w-[25rem] shrink-0 overflow-y-auto bg-db-cool-gray-200 p-2 dark:bg-gray-800">
        <TripList />
      </div>
      <div className="grow overflow-y-auto px-2">
        <Outlet />
      </div>
    </>
  );
}

export default TripsMainSection;
