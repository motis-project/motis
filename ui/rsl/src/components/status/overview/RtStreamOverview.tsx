import { formatDistanceToNow } from "date-fns";
import { de } from "date-fns/locale";
import React from "react";

import { RISSourceStatus } from "@/api/protocol/motis/ris.ts";

import { useRISStatusRequest } from "@/api/ris.ts";

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

function RisSourceStatus({
  name,
  status,
}: {
  name: string;
  status: RISSourceStatus | undefined;
}) {
  let icon = <StatusIconSkeleton />;
  let explanation = null;

  if (status) {
    const now = Date.now() / 1000;
    const maxDelay =
      status.update_interval !== 0 ? status.update_interval * 2 : 120;
    if (!status.enabled) {
      icon = <StatusIconError />;
      explanation = "Datenstrom ist nicht konfiguriert";
    } else if (status.last_update_time == 0) {
      icon = <StatusIconError />;
      explanation =
        "Datenstrom liefert keine Nachrichten (möglicherweise ungültige Konfiguration)";
    } else if (now - status.last_update_time > maxDelay) {
      icon = <StatusIconError />;
      explanation = `Nachrichten wurden zuletzt ${formatDistanceToNow(
        new Date(status.last_update_time * 1000),
        { locale: de, addSuffix: true },
      )} empfangen`;
    } else if (now - status.last_message_time > maxDelay) {
      icon = <StatusIconWarning />;
      explanation = `Datenstrom liefert Nachrichten, allerdings wurden erst Nachrichten bis ${formatDistanceToNow(
        new Date(status.last_message_time * 1000),
        { locale: de, addSuffix: true },
      )} empfangen`;
    } else {
      icon = <StatusIconOk />;
    }
  }

  return (
    <div>
      <div className="flex gap-2">
        {icon}
        <span className="font-semibold">{name}</span>
      </div>
      {explanation && <div className="ml-8">{explanation}</div>}
    </div>
  );
}

export function RtStreamOverview() {
  const { data: risStatus } = useRISStatusRequest();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">
          Empfang der Echtzeitdatenströme
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <RisSourceStatus
          name="RI Basis Fahrt (Echtzeitmeldungen)"
          status={risStatus?.ribasis_fahrt_status}
        />
        <RisSourceStatus
          name="RI Basis Formation (Wagenreihungen)"
          status={risStatus?.ribasis_formation_status}
        />
      </CardContent>
    </Card>
  );
}
