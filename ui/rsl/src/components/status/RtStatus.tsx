import { ReactElement } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { useRtMetricsRequest } from "@/api/rt";

import RtMetricsChart from "@/components/status/RtMetricsChart";

function RtStatus(): ReactElement {
  // TODO: if no rt messages have been processed, the api will return an error (rt::error::schedule_not_found)
  const { data } = useRtMetricsRequest();

  return (
    <>
      <div className="py-3">
        <h2 className="text-lg font-semibold">Echtzeitmeldungen</h2>
        <RtMetricsDisplay metrics={data?.by_msg_timestamp} />
      </div>
      <div className="py-3">
        <h2 className="text-lg font-semibold">Wagenreihungsmeldungen</h2>
        <FormationMetricsDisplay metrics={data?.by_msg_timestamp} />
      </div>
    </>
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
    <div className="h-72">
      <RtMetricsChart
        metricsData={metrics}
        metricsInfo={{
          full_trip_schedule_messages: {
            label: "Sollfahrten",
            color: "#1e40af",
          },
          full_trip_update_messages: {
            label: "Fahrt Updates",
            color: "#c2410c",
          },
        }}
      />
    </div>
  );
}

function FormationMetricsDisplay({
  metrics,
}: RtMetricsDisplayProps): ReactElement {
  if (!metrics) {
    return <div>Statistiken werden geladen...</div>;
  }

  return (
    <div className="h-72">
      <RtMetricsChart
        metricsData={metrics}
        metricsInfo={{
          formation_schedule_messages: {
            label: "Sollwagenreihungen",
            color: "#1e40af",
          },
          formation_preview_messages: {
            label: "Wagenreihungsänderungen",
            color: "#c2410c",
          },
        }}
      />
    </div>
  );
}

export default RtStatus;
