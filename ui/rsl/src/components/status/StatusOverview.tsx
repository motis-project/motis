import { ReactElement } from "react";

import CapacityStatus from "@/components/status/CapacityStatus";
import RISStatus from "@/components/status/RISStatus";
import RtStatus from "@/components/status/RtStatus";

function StatusOverview(): ReactElement {
  return (
    <div className="p-3 grow overflow-y-auto">
      <h1 className="text-xl font-semibold">MOTIS RSL Status</h1>
      <RISStatus />
      <RtStatus />
      <CapacityStatus />
    </div>
  );
}

export default StatusOverview;
