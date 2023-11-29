import React from "react";

import { PaxMonDatasetInfoResponse } from "@/api/protocol/motis/paxmon.ts";

import {
  StatusIconError,
  StatusIconOk,
  StatusIconSkeleton,
  StatusIconWarning,
} from "@/components/status/overview/icons.tsx";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card.tsx";

export interface DatasetInfoProps {
  datasetInfo: PaxMonDatasetInfoResponse | undefined;
}

function ScheduleStatus({ datasetInfo }: DatasetInfoProps) {
  let icon = <StatusIconSkeleton />;

  if (datasetInfo) {
    if (
      datasetInfo.schedule.station_count > 0 &&
      datasetInfo.schedule.trip_count > 0
    ) {
      icon = <StatusIconOk />;
    } else {
      icon = <StatusIconError />;
    }
  }

  return (
    <div>
      <div className="flex gap-2">
        {icon}
        <span className="font-semibold">Sollfahrplan</span>
      </div>
    </div>
  );
}

function JourneyFilesStatus({ datasetInfo }: DatasetInfoProps) {
  let icon = <StatusIconSkeleton />;
  let explanation = null;

  if (datasetInfo) {
    if (datasetInfo.journey_files.length == 0) {
      icon = <StatusIconError />;
      explanation = "Keine Reisekettendateien konfiguriert";
    } else if (
      !datasetInfo.journey_files.some((jfi) => jfi.matched_journeys > 0)
    ) {
      icon = <StatusIconError />;
      explanation =
        "Reisekettendateien sind konfiguriert, enthalten aber keine Reiseketten die zum Sollfahrplan passen.";
    } else if (
      !datasetInfo.journey_files.some(
        (jfi) =>
          jfi.matched_journeys > 0 &&
          jfi.matched_journeys /
            (jfi.matched_journeys + jfi.unmatched_journeys) >=
            0.7,
      )
    ) {
      icon = <StatusIconWarning />;
      explanation =
        "Reisekettendateien sind konfiguriert, aber nur wenige Reiseketten konnten geladen werden. Wahrscheinlich passen Sollfahrplan und Reisekettendateien nicht zusammen, z.B. weil der Sollfahrplan veraltet ist.";
    } else {
      icon = <StatusIconOk />;
    }
  }

  return (
    <div>
      <div className="flex gap-2">
        {icon}
        <span className="font-semibold">Reiseketten</span>
      </div>
      {explanation && <div className="ml-8">{explanation}</div>}
    </div>
  );
}

function CapacityFilesStatus({ datasetInfo }: DatasetInfoProps) {
  let icon = <StatusIconSkeleton />;
  let explanation = null;

  if (datasetInfo) {
    if (datasetInfo.capacity_files.length == 0) {
      icon = <StatusIconError />;
      explanation = "Keine Kapazitätsdateien konfiguriert";
    } else if (
      !datasetInfo.capacity_files.some((cfi) => cfi.loaded_entry_count > 0)
    ) {
      icon = <StatusIconError />;
      explanation =
        "Kapazitätsdateien sind konfiguriert, enthalten aber keine Daten, die verarbeitet werden konnten";
    } else {
      icon = <StatusIconOk />;
    }
  }

  return (
    <div>
      <div className="flex gap-2">
        {icon}
        <span className="font-semibold">Kapazitätsdaten</span>
      </div>
      {explanation && <div className="ml-8">{explanation}</div>}
    </div>
  );
}

export function StaticDatasetOverview({
  datasetInfo,
}: {
  datasetInfo: PaxMonDatasetInfoResponse | undefined;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Statische Datensätze</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <ScheduleStatus datasetInfo={datasetInfo} />
        <JourneyFilesStatus datasetInfo={datasetInfo} />
        <CapacityFilesStatus datasetInfo={datasetInfo} />
      </CardContent>
    </Card>
  );
}
