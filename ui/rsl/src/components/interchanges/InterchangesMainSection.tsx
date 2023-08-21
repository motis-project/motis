import { Outlet } from "react-router-dom";

import CriticalInterchangeList from "@/components/interchanges/CriticalInterchangeList";

function InterchangesMainSection(): JSX.Element {
  return (
    <>
      <div className="bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0">
        <CriticalInterchangeList />
      </div>
      <div className="overflow-y-auto grow p-2">
        <Outlet />
      </div>
    </>
  );
}

export default InterchangesMainSection;
