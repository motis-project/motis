import {
  QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useAtom } from "jotai";
import { useAtomCallback } from "jotai/utils";
import { Unplug } from "lucide-react";
import { useCallback } from "react";

import { PaxMonStatusResponse } from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonStatusRequest } from "@/api/paxmon";
import { sendRISForwardTimeRequest } from "@/api/ris";

import {
  defaultUniverse,
  multiverseIdAtom,
  scheduleAtom,
  universeAtom,
  universesAtom,
} from "@/data/multiverse";

import { formatDate, formatDateTime, formatTime } from "@/util/dateFormat";

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

async function forwardTimeByStepped(
  queryClient: QueryClient,
  schedule: number,
  currentTime: number,
  forwardBy: number,
  stepSize = 60,
) {
  const endTime = currentTime + forwardBy;
  while (currentTime < endTime) {
    currentTime = Math.min(endTime, currentTime + stepSize);
    await sendRISForwardTimeRequest(currentTime, schedule);
    await queryClient.invalidateQueries();
  }
  return currentTime;
}

interface TimeControlProps {
  allowForwarding: boolean;
}

function TimeControl({ allowForwarding }: TimeControlProps): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [schedule] = useAtom(scheduleAtom);

  const updateMultiverseId = useAtomCallback(
    useCallback((get, set, arg: number) => {
      const currentMultiverseId = get(multiverseIdAtom);
      if (currentMultiverseId != arg) {
        set(multiverseIdAtom, arg);
        if (currentMultiverseId != 0) {
          // multiverse id changed = server restarted -> reset universes
          console.log(
            `multiverse id changed: ${currentMultiverseId} -> ${arg}`,
          );
          set(universeAtom, 0);
          set(scheduleAtom, 0);
          set(universesAtom, [defaultUniverse]);
        }
      }
    }, []),
  );
  const {
    data: status,
    isLoading,
    error,
  } = useQuery(
    queryKeys.status(universe),
    () => sendPaxMonStatusRequest({ universe }),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      staleTime: 0,
      onSuccess: (data) => {
        updateMultiverseId(data.multiverse_id);
      },
    },
  );

  const forwardMutation = useMutation((forwardBy: number) => {
    return forwardTimeByStepped(
      queryClient,
      schedule,
      status?.system_time ?? 0,
      forwardBy,
    );
  });

  const forwardDisabled = forwardMutation.isLoading;

  const buttonClass = `px-3 py-1 rounded text-sm ${
    !forwardDisabled
      ? "bg-db-red-500 hover:bg-db-red-600 text-white"
      : "bg-db-red-300 text-db-red-100 cursor-wait"
  }`;

  const buttons = allowForwarding ? (
    <>
      {[1, 10, 30].map((min) => (
        <button
          key={`${min}m`}
          type="button"
          className={buttonClass}
          disabled={forwardDisabled}
          onClick={() => {
            forwardMutation.mutate(60 * min);
          }}
        >
          +{min}m
        </button>
      ))}
      {[1, 5, 10, 12, 24].map((hrs) => (
        <button
          key={`${hrs}h`}
          type="button"
          className={buttonClass}
          disabled={forwardDisabled}
          onClick={() => {
            forwardMutation.mutate(60 * 60 * hrs);
          }}
        >
          +{hrs}h
        </button>
      ))}
    </>
  ) : null;

  return (
    <div className="flex justify-center items-center space-x-2">
      {status ? (
        <>
          <HoverCard>
            <HoverCardTrigger asChild>
              <div className="flex justify-center items-center space-x-2">
                <SystemStatusIndicator status={status} />
                <div>{formatDate(status.system_time)}</div>
                <div className="font-bold">
                  {formatTime(status.system_time)}
                </div>
              </div>
            </HoverCardTrigger>
            <HoverCardContent className="w-96">
              <StatusHoverCardContent status={status} />
            </HoverCardContent>
          </HoverCard>
          {buttons}
        </>
      ) : isLoading ? (
        <div>Verbindung zu MOTIS wird aufgebaut...</div>
      ) : (
        <div>
          Fehler:{" "}
          {error instanceof Error
            ? error.message
            : `Verbindung zu MOTIS fehlgeschlagen.`}
        </div>
      )}
    </div>
  );
}

interface StatusProps {
  status: PaxMonStatusResponse;
}

function SystemStatusIndicator({ status }: StatusProps) {
  return (
    <StatusIndicator
      enabled={
        status.ribasis_fahrt_status.enabled ||
        status.ribasis_formation_status.enabled
      }
      receiving={
        status.ribasis_fahrt_status.receiving &&
        status.ribasis_formation_status.receiving
      }
      up_to_date={
        status.ribasis_fahrt_status.up_to_date &&
        status.ribasis_formation_status.up_to_date
      }
    />
  );
}

interface StatusIndicatorProps {
  enabled: boolean;
  receiving: boolean;
  up_to_date: boolean;
}

function StatusIndicator({
  enabled,
  receiving,
  up_to_date,
}: StatusIndicatorProps) {
  if (enabled) {
    if (!receiving) {
      return <div className="bg-red-500 rounded-full w-2 h-2"></div>;
    } else if (!up_to_date) {
      return <div className="bg-orange-500 rounded-full w-2 h-2"></div>;
    } else {
      return <div className="bg-green-500 rounded-full w-2 h-2"></div>;
    }
  } else {
    return <Unplug className="w-4 h-4 stroke-gray-700" />;
  }
}

function StatusHoverCardContent({ status }: StatusProps) {
  const rtEnabled =
    status.ribasis_fahrt_status || status.ribasis_formation_status;
  return (
    <div>
      <div className="font-semibold pb-2">Status der Echtzeitdatenströme:</div>
      <table className="w-full">
        <tbody>
          <tr>
            <td className="pr-2">
              <StatusIndicator {...status.ribasis_fahrt_status} />
            </td>
            <td className="pr-2">Echtzeitmeldungen</td>
            <td className="text-right">
              <Timestamp
                timestamp={status.ribasis_fahrt_status.last_message_time}
              />
            </td>
          </tr>
          <tr>
            <td className="pr-2">
              <StatusIndicator {...status.ribasis_formation_status} />
            </td>
            <td className="pr-2">Wagenreihungen</td>
            <td className="text-right">
              <Timestamp
                timestamp={status.ribasis_formation_status.last_message_time}
              />
            </td>
          </tr>
        </tbody>
      </table>
      {!rtEnabled && (
        <div className="pt-4">
          Es sind keine Echtzeitdatenströme konfiguriert.
        </div>
      )}
    </div>
  );
}

interface TimestampProps {
  timestamp: number;
}

function Timestamp({ timestamp }: TimestampProps) {
  if (timestamp === 0) {
    return <>Nicht verfügbar</>;
  } else {
    return <>{formatDateTime(timestamp)}</>;
  }
}

export default TimeControl;
