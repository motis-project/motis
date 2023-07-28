import { ReactElement } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { useRtMetricsRequest } from "@/api/rt";

import RtMetricsChart from "@/components/status/RtMetricsChart";

function RtStatus(): ReactElement {
  const { data } = useRtMetricsRequest();

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Echtzeitmeldungen</h2>
      <RtMetricsDisplay metrics={data?.by_msg_timestamp} />
    </div>
  );
}

interface RtMetricsDisplayProps {
  metrics: RtMetrics | undefined;
}

function RtMetricsDisplay({ metrics }: RtMetricsDisplayProps): ReactElement {
  if (!metrics) {
    return <div>Statistiken werden geladen...</div>;
  }
  return (
    <div className="h-96">
      <RtMetricsChart metrics={metrics} />
    </div>
  );
}

export default RtStatus;
