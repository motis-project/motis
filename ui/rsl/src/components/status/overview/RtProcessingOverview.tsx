import React, { ReactElement } from "react";

import { RISStatusResponse } from "@/api/protocol/motis/ris.ts";
import { RtMetricsResponse } from "@/api/protocol/motis/rt.ts";

import { useRISStatusRequest } from "@/api/ris.ts";
import { useRtMetricsRequest } from "@/api/rt.ts";

import { formatPercent } from "@/data/numberFormat.ts";

import {
  StatusIconError,
  StatusIconOk,
  StatusIconWarning,
} from "@/components/status/overview/icons.tsx";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card.tsx";

interface RtStatusProps {
  rtMetrics: RtMetricsResponse;
  risStatus: RISStatusResponse;
}

function FahrtStatus({ rtMetrics, risStatus }: RtStatusProps) {
  const metrics = rtMetrics.by_msg_timestamp;
  const last_timestamp = Math.max(
    risStatus.init_status.last_message_time,
    risStatus.ribasis_fahrt_status.last_message_time,
  );
  const last_idx = Math.floor((last_timestamp - metrics.start_time) / 60);

  if (last_idx < 0 || last_idx >= metrics.entries) {
    return (
      <div>
        Keine Informationen zur Verarbeitung von Echtzeitmeldungen vorhanden.
      </div>
    );
  }

  const interval_start = Math.max(0, last_idx - 59);
  const interval_length = last_idx - interval_start + 1;

  let schedule_msgs = 0;
  let update_msgs = 0;
  let trip_id_not_found = 0;
  let trip_id_ambiguous = 0;
  for (let i = interval_start; i <= last_idx; i++) {
    schedule_msgs += metrics.full_trip_schedule_messages[i];
    update_msgs += metrics.full_trip_update_messages[i];
    trip_id_not_found += metrics.trip_id_not_found[i];
    trip_id_ambiguous += metrics.trip_id_ambiguous[i];
  }

  const total_msgs = schedule_msgs + update_msgs;
  const trip_id_error_rate =
    total_msgs > 0 ? (trip_id_not_found + trip_id_ambiguous) / total_msgs : 0;

  const messages: ReactElement[] = [];

  if (total_msgs === 0) {
    messages.push(
      <div className="flex gap-2" key="no-messages">
        <StatusIconError />
        Keine Nachrichten empfangen.
      </div>,
    );
  } else if (update_msgs === 0) {
    messages.push(
      <div className="flex gap-2" key="no-update-messages">
        <StatusIconError />
        Der Datenstrom liefert nur Sollfahrten, keine Echtzeitupdates.
      </div>,
    );
  }

  if (total_msgs != 0) {
    const error_rate_text = `${formatPercent(trip_id_error_rate, {
      minimumFractionDigits: 2,
    })} der Züge konnten nicht zugeordnet werden.`;

    if (trip_id_error_rate <= 0.01) {
      messages.push(
        <div className="flex gap-2" key="ok">
          <StatusIconOk />
          <div>
            <p className="font-semibold">
              Nahezu alle Echtzeitmeldungen konnten verarbeitet werden.
            </p>
            <p>{error_rate_text}</p>
          </div>
        </div>,
      );
    } else if (trip_id_error_rate <= 0.1) {
      messages.push(
        <div className="flex gap-2" key="some-errors">
          <StatusIconWarning />
          <div>
            <p className="font-semibold">
              Einige Echtzeitmeldungen konnten nicht verarbeitet werden.
            </p>
            <p>{error_rate_text} </p>
            <p>
              Möglicherweise passen der Sollfahrplan und der Echtzeitdatenstrom
              nicht zusammen, z.B. weil der Sollfahrplan veraltet ist.
            </p>
          </div>
        </div>,
      );
    } else {
      messages.push(
        <div className="flex gap-2" key="many-errors">
          <StatusIconError />
          <div>
            <p className="font-semibold">
              Viele Echtzeitmeldungen konnten nicht verarbeitet werden.
            </p>
            <p>{error_rate_text} </p>
            <p>
              Wahrscheinlich passen der Sollfahrplan und der Echtzeitdatenstrom
              nicht zusammen, z.B. weil der Sollfahrplan veraltet ist.
            </p>
          </div>
        </div>,
      );
    }
  }

  return (
    <>
      <div>In den letzten {interval_length} Minuten:</div>
      {messages}
    </>
  );
}

function FormationStatus({ rtMetrics, risStatus }: RtStatusProps) {
  const metrics = rtMetrics.by_msg_timestamp;
  const last_timestamp = Math.max(
    risStatus.init_status.last_message_time,
    risStatus.ribasis_fahrt_status.last_message_time,
  );
  const last_idx = Math.floor((last_timestamp - metrics.start_time) / 60);

  if (last_idx < 0 || last_idx >= metrics.entries) {
    return (
      <div>
        Keine Informationen zur Verarbeitung von Wagenreihungsmeldungen
        vorhanden.
      </div>
    );
  }

  const interval_start = Math.max(0, last_idx - 59);
  const interval_length = last_idx - interval_start + 1;

  let total_msgs = 0;
  let invalid_trip_id = 0;
  let trip_id_not_found = 0;
  let trip_id_ambiguous = 0;
  for (let i = interval_start; i <= last_idx; i++) {
    total_msgs += metrics.formation_schedule_messages[i];
    total_msgs += metrics.formation_preview_messages[i];
    total_msgs += metrics.formation_is_messages[i];
    invalid_trip_id += metrics.formation_invalid_trip_id[i];
    trip_id_not_found += metrics.formation_trip_id_not_found[i];
    trip_id_ambiguous += metrics.formation_trip_id_ambiguous[i];
  }

  const trip_id_error_rate =
    total_msgs > 0
      ? (invalid_trip_id + trip_id_not_found + trip_id_ambiguous) / total_msgs
      : 0;

  const messages: ReactElement[] = [];

  if (total_msgs === 0) {
    messages.push(
      <div className="flex gap-2" key="no-messages">
        <StatusIconError />
        Keine Nachrichten empfangen.
      </div>,
    );
  }

  if (total_msgs != 0) {
    const error_rate_text = `${formatPercent(trip_id_error_rate, {
      minimumFractionDigits: 2,
    })} der Züge konnten nicht zugeordnet werden.`;

    if (trip_id_error_rate <= 0.01) {
      messages.push(
        <div className="flex gap-2" key="ok">
          <StatusIconOk />
          <div>
            <p className="font-semibold">
              Nahezu alle Wagenreihungsmeldungen konnten verarbeitet werden.
            </p>
            <p>{error_rate_text}</p>
          </div>
        </div>,
      );
    } else if (trip_id_error_rate <= 0.1) {
      messages.push(
        <div className="flex gap-2" key="some-errors">
          <StatusIconWarning />
          <div>
            <p className="font-semibold">
              Einige Wagenreihungsmeldungen konnten nicht verarbeitet werden.
            </p>
            <p>{error_rate_text} </p>
            <p>
              Möglicherweise passen der Sollfahrplan und der Echtzeitdatenstrom
              nicht zusammen, z.B. weil der Sollfahrplan veraltet ist.
            </p>
          </div>
        </div>,
      );
    } else {
      messages.push(
        <div className="flex gap-2" key="many-errors">
          <StatusIconError />
          <div>
            <p className="font-semibold">
              Viele Wagenreihungsmeldungen konnten nicht verarbeitet werden.
            </p>
            <p>{error_rate_text} </p>
            <p>
              Wahrscheinlich passen der Sollfahrplan und der Echtzeitdatenstrom
              nicht zusammen, z.B. weil der Sollfahrplan veraltet ist.
            </p>
          </div>
        </div>,
      );
    }
  }

  return (
    <>
      <div>In den letzten {interval_length} Minuten:</div>
      {messages}
    </>
  );
}

export function RtProcessingOverview() {
  const { data: rtMetrics, isError } = useRtMetricsRequest();
  const { data: risStatus } = useRISStatusRequest();

  return (
    <div className="mt-4 grid grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Verarbeitung der Echtzeitmeldungen
          </CardTitle>
        </CardHeader>
        {isError ? (
          <CardContent>Keine Echtzeitdaten vorhanden.</CardContent>
        ) : rtMetrics && risStatus ? (
          <CardContent className="flex flex-col gap-2">
            <FahrtStatus rtMetrics={rtMetrics} risStatus={risStatus} />
          </CardContent>
        ) : (
          <CardContent>Status wird abgefragt...</CardContent>
        )}
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Verarbeitung der Wagenreihungsmeldungen
          </CardTitle>
        </CardHeader>
        {isError ? (
          <CardContent>Keine Echtzeitdaten vorhanden.</CardContent>
        ) : rtMetrics && risStatus ? (
          <CardContent className="flex flex-col gap-2">
            <FormationStatus rtMetrics={rtMetrics} risStatus={risStatus} />
          </CardContent>
        ) : (
          <CardContent>Status wird abgefragt...</CardContent>
        )}
      </Card>
    </div>
  );
}
