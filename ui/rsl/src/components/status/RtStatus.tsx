import { ReactElement, useState } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { useRISStatusRequest } from "@/api/ris";
import { useRtMetricsRequest } from "@/api/rt";

import MetricsChart from "@/components/status/MetricsChart";
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

  const { data: risStatus } = useRISStatusRequest();
  const hideUntil =
    grouping === "by_processing_time"
      ? risStatus?.init_status?.last_update_time ?? 0
      : 0;

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
        <RtMetricsDisplay
          metrics={data ? data[grouping] : undefined}
          hideUntil={hideUntil}
        />
      </div>
      <div className="py-3">
        <h2 className="text-lg font-semibold">Wagenreihungsmeldungen</h2>
        <FormationMetricsDisplay
          metrics={data ? data[grouping] : undefined}
          hideUntil={hideUntil}
        />
      </div>
    </div>
  );
}

interface RtMetricsDisplayProps {
  metrics: RtMetrics | undefined;
  hideUntil: number;
}

function RtMetricsDisplay({
  metrics,
  hideUntil,
}: RtMetricsDisplayProps): ReactElement {
  if (!metrics) {
    return <div>Statistiken werden geladen...</div>;
  }

  return (
    <div>
      <h3 className="font-semibold">
        Empfangene Echtzeitnachrichten pro Minute
      </h3>
      <div className="h-72 py-2">
        <MetricsChart
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
          hideUntil={hideUntil}
        />
      </div>
      <h3 className="font-semibold">
        Verarbeitung der empfangenen Echtzeitnachrichten
      </h3>
      <div className="h-72 py-2">
        <MetricsChart
          metricsData={metrics}
          metricsInfo={{
            trip_id_not_found: {
              label: "Zug nicht gefunden",
              color: "#1e40af",
            },
            trip_id_ambiguous: {
              label: "Zug nicht eindeutig",
              color: "#c2410c",
            },
          }}
          hideUntil={hideUntil}
        />
      </div>
      <div className="h-72 py-2">
        <MetricsChart
          metricsData={metrics}
          metricsInfo={{
            trip_delay_updates: {
              label: "Verarbeitete Verspätungsupdates (Anzahl Züge)",
              color: "#1e40af",
            },
          }}
          hideUntil={hideUntil}
        />
      </div>
      <div className="h-72 py-2">
        <MetricsChart
          metricsData={metrics}
          metricsInfo={{
            reroutes: {
              label: "Verarbeitete Umleitungen",
              color: "#700cc2",
            },
            rule_service_reroutes: {
              label: "Umleitungen vereinigter/durchgebundener Züge",
              color: "#c20c0c",
            },
          }}
          hideUntil={hideUntil}
        />
      </div>
      <div className="h-72 py-2">
        <MetricsChart
          metricsData={metrics}
          metricsInfo={{
            trip_track_updates: {
              label: "Verarbeitete Gleisänderungen",
              color: "#0f6105",
            },
          }}
          hideUntil={hideUntil}
        />
      </div>
    </div>
  );
}

function FormationMetricsDisplay({
  metrics,
  hideUntil,
}: RtMetricsDisplayProps): ReactElement {
  if (!metrics) {
    return <div>Statistiken werden geladen...</div>;
  }

  return (
    <div>
      <h3 className="font-semibold">
        Empfangene Wagenreihungsmeldungen pro Minute
      </h3>
      <div className="h-72 py-2">
        <MetricsChart
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
          hideUntil={hideUntil}
        />
      </div>
      <h3 className="font-semibold">Verarbeitung der Wagenreihungsmeldungen</h3>
      <div className="h-72 py-2">
        <MetricsChart
          metricsData={metrics}
          metricsInfo={{
            formation_trip_id_not_found: {
              label: "Zug nicht gefunden",
              color: "#1e40af",
            },
            formation_trip_id_ambiguous: {
              label: "Zug nicht eindeutig",
              color: "#c2410c",
            },
            formation_invalid_trip_id: {
              label: "Ungültige Zug ID",
              color: "#700cc2",
            },
          }}
          hideUntil={hideUntil}
        />
      </div>
    </div>
  );
}

export default RtStatus;
