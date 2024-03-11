import { formatDistanceToNow } from "date-fns";
import { de } from "date-fns/locale";
import { History, MonitorPlay, TrainFront, Users } from "lucide-react";
import React, { ReactNode } from "react";

import { usePaxMonDatasetInfo, usePaxMonStatusQuery } from "@/api/paxmon.ts";

import { formatNumber, formatPercent } from "@/data/numberFormat.ts";

import { formatDateTime } from "@/util/dateFormat.ts";

import { RtProcessingOverview } from "@/components/status/overview/RtProcessingOverview.tsx";
import { RtStreamOverview } from "@/components/status/overview/RtStreamOverview.tsx";
import { StaticDatasetOverview } from "@/components/status/overview/StaticDatasetOverview.tsx";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card.tsx";
import { Skeleton } from "@/components/ui/skeleton.tsx";

function StatusOverview(): ReactNode {
  const { data: paxmonStatus } = usePaxMonStatusQuery(0);
  const { data: datasetInfo } = usePaxMonDatasetInfo();

  return (
    <div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Überwachte Reisendengruppen
            </CardTitle>
            <Users
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
          </CardHeader>
          <CardContent>
            {paxmonStatus ? (
              <>
                <div className="text-2xl font-bold">
                  {formatNumber(paxmonStatus.active_groups)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {`${formatNumber(paxmonStatus.active_pax)} Reisende`}
                </div>
              </>
            ) : (
              <>
                <Skeleton className="h-8 w-24" />
                <Skeleton className="h-4 w-48" />
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Überwachte Züge
            </CardTitle>
            <TrainFront
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {paxmonStatus ? (
                formatNumber(paxmonStatus.trip_count)
              ) : (
                <Skeleton className="h-8 w-24" />
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              {paxmonStatus &&
              datasetInfo &&
              datasetInfo.schedule.trip_count > 0 ? (
                `von ${formatNumber(
                  datasetInfo.schedule.trip_count,
                )} Zügen aus dem Fahrplan (${formatPercent(
                  paxmonStatus.trip_count / datasetInfo.schedule.trip_count,
                )})`
              ) : (
                <Skeleton className="h-4 w-48" />
              )}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Echtzeitverzögerung
            </CardTitle>
            <History
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
          </CardHeader>
          <CardContent>
            {paxmonStatus ? (
              <>
                <div className="text-2xl font-bold">
                  {formatDistanceToNow(
                    new Date(paxmonStatus.primary_system_time * 1000),
                    { locale: de },
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  {paxmonStatus.ribasis_fahrt_status.receiving
                    ? `Nachrichten werden alle ${paxmonStatus.ribasis_fahrt_status.update_interval} Sekunden verarbeitet`
                    : paxmonStatus.ribasis_fahrt_status.enabled
                      ? "Echtzeitdatenstrom nicht aktiviert"
                      : "Empfang der Echtzeitdaten gestört"}
                </div>
              </>
            ) : (
              <>
                <Skeleton className="h-8 w-24" />
                <Skeleton className="h-4 w-48" />
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Letzter MOTIS-Neustart
            </CardTitle>
            <MonitorPlay
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
          </CardHeader>
          <CardContent>
            {paxmonStatus ? (
              <>
                <div className="text-2xl font-bold">
                  {formatDistanceToNow(
                    new Date(paxmonStatus.motis_start_time * 1000),
                    { locale: de, addSuffix: true },
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  {formatDateTime(paxmonStatus.motis_start_time)}
                </div>
              </>
            ) : (
              <>
                <Skeleton className="h-8 w-24" />
                <Skeleton className="h-4 w-48" />
              </>
            )}
          </CardContent>
        </Card>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <StaticDatasetOverview datasetInfo={datasetInfo} />
        <RtStreamOverview />
      </div>
      <div className="mt-4">
        <RtProcessingOverview />
      </div>
    </div>
  );
}

export default StatusOverview;
