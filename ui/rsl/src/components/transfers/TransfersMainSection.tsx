import { ReactElement } from "react";
import { Outlet } from "react-router-dom";

import BrokenTransfersList from "@/components/transfers/BrokenTransfersList";

function TransfersMainSection(): ReactElement {
  return (
    <>
      <div className="w-[25rem] shrink-0 overflow-y-auto bg-db-cool-gray-200 p-2 dark:bg-gray-800">
        <BrokenTransfersList />
      </div>
      <div className="grow overflow-y-auto p-2">
        <Outlet />
      </div>
    </>
  );
}

export default TransfersMainSection;
