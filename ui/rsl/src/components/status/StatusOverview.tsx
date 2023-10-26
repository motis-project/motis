import { ReactElement } from "react";

import CapacityStatus from "@/components/status/CapacityStatus";
import RISStatus from "@/components/status/RISStatus";
import RslStatus from "@/components/status/RslStatus";
import RtStatus from "@/components/status/RtStatus";

function StatusOverview(): ReactElement {
  return (
    <div className="grow overflow-y-auto p-3">
      <h1 className="text-xl font-semibold">MOTIS RSL Status</h1>
      <RISStatus />
      <RtStatus />
      <RslStatus />
      <CapacityStatus />
    </div>
  );
}

export default StatusOverview;
