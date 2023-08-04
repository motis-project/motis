import { useQuery } from "@tanstack/react-query";
import { ReactElement, useState } from "react";

import {
  PaxForecastMetricsRequest,
  PaxForecastMetricsResponse,
} from "@/api/protocol/motis/paxforecast";
import {
  PaxMonMetricsRequest,
  PaxMonMetricsResponse,
} from "@/api/protocol/motis/paxmon";

import {
  queryKeys as paxForecastQueryKeys,
  sendPaxForecastMetricsRequest,
} from "@/api/paxforecast";
import {
  queryKeys as paxmonQueryKeys,
  sendPaxMonMetricsRequest,
} from "@/api/paxmon";
import { useRISStatusRequest } from "@/api/ris";

import MetricsChart from "@/components/status/MetricsChart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type MetricsGrouping = "by_system_time" | "by_processing_time";

type MetricsApi = "paxmon" | "paxforecast";

interface MetricsOption {
  api: MetricsApi;
  key: string;
  label: string;
}

const metricsOptions: MetricsOption[] = [
  {
    api: "paxmon",
    key: "affected_group_routes",
    label: "Betroffene Reisendengruppen",
  },
  {
    api: "paxmon",
    key: "broken_group_routes",
    label: "Gebrochene Reisendengruppen",
  },
  {
    api: "paxmon",
    key: "major_delay_group_routes",
    label: "Reisendengruppen mit hoher Zielverspätung",
  },
  {
    api: "paxforecast",
    key: "major_delay_group_routes_with_alternatives",
    label: "Reisendengruppen mit hoher Zielverspätung und Alternativen",
  },
  {
    api: "paxforecast",
    key: "rerouted_group_routes",
    label: "Umgeleitete Reisendengruppen",
  },
  {
    api: "paxforecast",
    key: "routing_requests",
    label: "Routing-Anfragen",
  },
  {
    api: "paxforecast",
    key: "alternatives_found",
    label: "Gefundene Alternativen",
  },
  {
    api: "paxmon",
    key: "total_timing",
    label: "Gesamt-Verarbeitungszeit (ms)",
  },
];

function RslStatus(): ReactElement {
  const [selectedMetric, setSelectedMetric] = useState<MetricsOption>(
    metricsOptions[0],
  );
  const [grouping, setGrouping] = useState<MetricsGrouping>("by_system_time");

  const paxmonRequest: PaxMonMetricsRequest = { universe: 0 };
  const { data: paxmonData } = useQuery(
    paxmonQueryKeys.metrics(paxmonRequest),
    () => sendPaxMonMetricsRequest(paxmonRequest),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  );

  const paxforecastRequest: PaxForecastMetricsRequest = { universe: 0 };
  const { data: paxforecastData } = useQuery(
    paxForecastQueryKeys.metrics(paxforecastRequest),
    () => sendPaxForecastMetricsRequest(paxforecastRequest),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  );

  const { data: risStatus } = useRISStatusRequest();
  const hideUntil = risStatus?.init_status?.last_update_time ?? 0;

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">RSL</h2>
      <div className="pt-2 flex justify-between">
        <Select
          value={selectedMetric.label}
          onValueChange={(v) => {
            const metric = metricsOptions.find((m) => m.label === v);
            if (metric) {
              setSelectedMetric(metric);
            } else {
              console.log(`Invalid RSL metric selected: "${metric}"`);
            }
          }}
        >
          <SelectTrigger className="w-[450px]">
            <SelectValue placeholder="Metrik wählen..." />
          </SelectTrigger>
          <SelectContent>
            {metricsOptions.map((option) => (
              <SelectItem value={option.label} key={option.label}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={grouping}
          onValueChange={(v) => setGrouping(v as MetricsGrouping)}
        >
          <SelectTrigger className="w-[300px]">
            <SelectValue placeholder="Statistiken gruppieren nach..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="by_system_time">Nach Universumszeit</SelectItem>
            <SelectItem value="by_processing_time">
              Nach Verarbeitungszeit
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="py-3">
        <RslMetricsDisplay
          paxmonData={paxmonData}
          paxforecastData={paxforecastData}
          grouping={grouping}
          option={selectedMetric}
          hideUntil={hideUntil}
        />
      </div>
    </div>
  );
}

interface RslMetricsDisplayProps {
  paxmonData: PaxMonMetricsResponse | undefined;
  paxforecastData: PaxForecastMetricsResponse | undefined;
  grouping: MetricsGrouping;
  option: MetricsOption;
  hideUntil: number;
}

function RslMetricsDisplay({
  paxmonData,
  paxforecastData,
  grouping,
  option,
  hideUntil,
}: RslMetricsDisplayProps): ReactElement {
  const data = option.api === "paxmon" ? paxmonData : paxforecastData;

  if (!data) {
    return <div>Statistiken werden geladen...</div>;
  }

  const metrics = data[grouping];

  return (
    <div className="h-72">
      <MetricsChart
        metricsData={metrics}
        metricsInfo={{
          [option.key]: {
            label: option.label,
            color: "#1e40af",
          },
        }}
        hideUntil={hideUntil}
      />
    </div>
  );
}

export default RslStatus;
