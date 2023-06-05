import { ReactElement } from "react";

import RISStatus from "@/components/status/RISStatus";

function StatusOverview(): ReactElement {
  return (
    <div className="p-3 grow overflow-y-auto">
      <h1 className="text-xl font-semibold">MOTIS RSL Status</h1>
      <RISStatus />
    </div>
  );
}

export default StatusOverview;
