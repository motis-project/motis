import { ReactElement } from "react";

import { RISSourceStatus, RISStatusResponse } from "@/api/protocol/motis/ris";

import { useRISStatusRequest } from "@/api/ris";

import { formatNumber } from "@/data/numberFormat";

import { formatDateTime } from "@/util/dateFormat";

function RISStatus(): ReactElement {
  const { data } = useRISStatusRequest();

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Echtzeitdatenströme</h2>
      <RISStatusDisplay data={data} />
    </div>
  );
}

function RISStatusDisplay({
  data,
}: {
  data: RISStatusResponse | undefined;
}): ReactElement {
  if (!data) {
    return <div>Status wird geladen...</div>;
  }
  return (
    <div>
      <p>
        Letzte Echtzeitmeldung: {formatDateTime(data.system_time)}, empfangen:{" "}
        {formatDateTime(data.last_update_time)}
      </p>
      <table className="border-separate border-spacing-x-2">
        <thead>
          <tr className="text-left">
            <th className="font-medium">Datenstrom</th>
            <th className="font-medium">Letzte Echtzeitmeldung</th>
            <th className="font-medium">Letzte Aktualisierung</th>
            <th className="font-medium">Nachrichten insgesamt</th>
          </tr>
        </thead>
        <tbody>
          <SourceStatus
            name="RI Basis Fahrt"
            status={data.ribasis_fahrt_status}
            hideIfDisabled={false}
            checkDelay={true}
          />
          <SourceStatus
            name="RI Basis Formation"
            status={data.ribasis_formation_status}
            hideIfDisabled={false}
            checkDelay={true}
          />
          <SourceStatus
            name="GTFS-RT"
            status={data.gtfs_rt_status}
            hideIfDisabled={true}
            checkDelay={true}
          />
          <SourceStatus
            name="Upload"
            status={data.upload_status}
            hideIfDisabled={true}
            checkDelay={true}
          />
          <SourceStatus
            name="Dateisystem"
            status={data.read_status}
            hideIfDisabled={true}
            checkDelay={true}
          />
          <SourceStatus
            name="Initialisierung aus Datenbank"
            status={data.init_status}
            hideIfDisabled={false}
            checkDelay={false}
          />
        </tbody>
      </table>
    </div>
  );
}

interface SourceStatusProps {
  name: string;
  status: RISSourceStatus;
  hideIfDisabled: boolean;
  checkDelay: boolean;
}

function SourceStatus({
  name,
  status,
  hideIfDisabled,
  checkDelay,
}: SourceStatusProps): ReactElement | null {
  if (!status.enabled) {
    if (hideIfDisabled) {
      return null;
    } else {
      return (
        <tr>
          <td className="font-medium">{name}</td>
          <td colSpan={3} className="text-center italic">
            — deaktiviert —
          </td>
        </tr>
      );
    }
  }

  const maxDelay =
    status.update_interval !== 0 ? status.update_interval * 2 : 120;

  return (
    <tr>
      <td className="font-medium">{name}</td>
      <td>{formatTS(status.last_message_time, checkDelay, maxDelay)}</td>
      <td>
        {formatTS(status.last_update_time, checkDelay, maxDelay)} (
        {formatNumber(status.last_update_messages)} Nachrichten)
      </td>
      <td>{formatNumber(status.total_messages)}</td>
    </tr>
  );
}

function formatTS(
  timestamp: number,
  colored: boolean,
  maxDelay = 120,
): ReactElement {
  if (timestamp === 0) {
    return <>—</>;
  }
  const delay = Date.now() / 1000 - timestamp;
  if (!colored || delay < maxDelay) {
    return <>{formatDateTime(timestamp)}</>;
  } else {
    return <span className="text-red-500">{formatDateTime(timestamp)}</span>;
  }
}

export default RISStatus;
