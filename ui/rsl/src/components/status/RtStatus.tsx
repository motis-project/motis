import { ReactElement, useState } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { useRtMetricsRequest } from "@/api/rt";

import RtMetricsChart from "@/components/status/RtMetricsChart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type MetricsGrouping = "by_msg_timestamp" | "by_processing_time";

function RtStatus(): ReactElement {
  const [grouping, setGrouping] = useState<MetricsGrouping>("by_msg_timestamp");
  const { data, isError } = useRtMetricsRequest();

  if (isError) {
    return <div>Keine Echtzeitdaten vorhanden.</div>;
  }

  return (
    <div>
      <div className="flex justify-end">
        <Select
          value={grouping}
          onValueChange={(v) => setGrouping(v as MetricsGrouping)}
        >
          <SelectTrigger className="w-[300px]">
            <SelectValue placeholder="Statistiken gruppieren nach..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="by_msg_timestamp">
              Nach Nachrichtenzeitstempel
            </SelectItem>
            <SelectItem value="by_processing_time">
              Nach Verarbeitungszeit
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="py-3">
        <h2 className="text-lg font-semibold">Echtzeitmeldungen</h2>
        <RtMetricsDisplay metrics={data ? data[grouping] : undefined} />
      </div>
      <div className="py-3">
        <h2 className="text-lg font-semibold">Wagenreihungsmeldungen</h2>
        <FormationMetricsDisplay metrics={data ? data[grouping] : undefined} />
      </div>
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
