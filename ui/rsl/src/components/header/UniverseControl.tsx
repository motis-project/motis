import { ArrowDownIcon } from "@heroicons/react/20/solid";
import { useIsMutating, useMutation, useQuery } from "@tanstack/react-query";
import { useAtom } from "jotai";
import { useAtomCallback } from "jotai/utils";
import { useCallback } from "react";

import {
  PaxMonGetUniversesResponse,
  PaxMonKeepAliveRequest,
  PaxMonKeepAliveResponse,
} from "@/api/protocol/motis/paxmon";

import {
  queryKeys,
  sendPaxMonDestroyUniverseRequest,
  sendPaxMonForkUniverseRequest,
  sendPaxMonGetUniversesRequest,
  sendPaxMonKeepAliveRequest,
} from "@/api/paxmon";

import {
  UniverseInfo,
  multiverseIdAtom,
  scheduleAtom,
  universeAtom,
  universesAtom,
} from "@/data/multiverse";

function UniverseControl(): JSX.Element {
  const [universe, setUniverse] = useAtom(universeAtom);
  const [schedule, setSchedule] = useAtom(scheduleAtom);
  const [multiverseId] = useAtom(multiverseIdAtom);
  const [universes, setUniverses] = useAtom(universesAtom);
  const isMutating = useIsMutating() != 0;

  const switchTo = useCallback(
    (uv: UniverseInfo) => {
      setUniverse(uv.id);
      setSchedule(uv.schedule);
    },
    [setUniverse, setSchedule],
  );

  const keepAliveRequest: PaxMonKeepAliveRequest = {
    multiverse_id: multiverseId,
    universes: universes.map((uv) => uv.id),
  };
  const keepAliveHandler = useAtomCallback(
    useCallback((get, set, data: PaxMonKeepAliveResponse) => {
      if (data.expired.length > 0) {
        console.log(`universes expired: ${data.expired.join(", ")}`);
        const currentUniverse = get(universeAtom);
        const newUniverses = get(universesAtom).filter(
          (uv) => uv.id === 0 || !data.expired.includes(uv.id),
        );
        if (currentUniverse !== 0 && data.expired.includes(currentUniverse)) {
          set(universeAtom, 0);
          set(scheduleAtom, 0);
        }
        set(universesAtom, newUniverses);
      }
    }, []),
  );
  useQuery(
    queryKeys.keepAlive(keepAliveRequest),
    () => sendPaxMonKeepAliveRequest(keepAliveRequest),
    {
      refetchInterval: 30 * 1000,
      refetchOnWindowFocus: true,
      refetchIntervalInBackground: true,
      notifyOnChangeProps: [],
      onSuccess: keepAliveHandler,
    },
  );

  const forkMutation = useMutation(
    (baseUniverse: number) =>
      sendPaxMonForkUniverseRequest({
        universe: baseUniverse,
        fork_schedule: true,
        ttl: 120,
      }),
    {
      onSuccess: (data) => {
        const newUv: UniverseInfo = {
          id: data.universe,
          schedule: data.schedule,
          ttl: data.ttl,
        };
        setUniverses([
          ...universes.filter((uv) => uv.id !== data.universe),
          newUv,
        ]);
        switchTo(newUv);
      },
    },
  );

  const destroyMutation = useMutation(
    (uv: number) => sendPaxMonDestroyUniverseRequest({ universe: uv }),
    {
      onSettled: (data, error, variables) => {
        if (error) {
          console.log(
            `error while trying to destroy universe ${variables}:`,
            error,
          );
        }
        setUniverses(universes.filter((u) => u.id != variables));
        switchTo(universes[0]);
      },
    },
  );

  const requestFromServerHandler = useAtomCallback(
    useCallback((get, set, data: PaxMonGetUniversesResponse) => {
      if (data.universes.length === 0) {
        console.log("error: server didn't return any universes");
        return;
      }
      const currentUniverse = get(universeAtom);
      const newUniverses: UniverseInfo[] = data.universes.map((uvi) => {
        return { id: uvi.universe, schedule: uvi.schedule, ttl: uvi.ttl };
      });
      set(multiverseIdAtom, data.multiverse_id);
      set(universesAtom, newUniverses);
      if (!newUniverses.find((uvi) => uvi.id === currentUniverse)) {
        set(universeAtom, newUniverses[0].id);
        set(scheduleAtom, newUniverses[0].schedule);
      }
    }, []),
  );

  const requestFromServerMutation = useMutation(sendPaxMonGetUniversesRequest, {
    onSettled: (data, error) => {
      if (error) {
        console.log(
          "error while trying to load list of universes from server:",
          error,
        );
      } else if (data) {
        requestFromServerHandler(data);
      }
    },
  });

  const forkEnabled = !isMutating;
  const destroyEnabled = !isMutating && universe != 0;
  const requestFromServerEnabled = !isMutating;

  // <PlusCircleIcon className="h-5 w-5 text-white" />
  // <XCircleIcon className="h-5 w-5 text-white" />

  return (
    <div className="flex justify-center items-center space-x-2 pl-4">
      <span className="pr-2">
        Universum #{universe} (Fahrplan #{schedule})
      </span>
      <button
        type="button"
        className={`inline-flex items-baseline px-3 py-1 rounded text-sm ${
          forkEnabled
            ? "bg-db-red-500 hover:bg-db-red-600 text-white"
            : "bg-db-red-300 text-db-red-100 cursor-default"
        }`}
        onClick={() => forkMutation.mutate(universe)}
        disabled={!forkEnabled}
      >
        Kopieren
      </button>
      <button
        type="button"
        className={`px-3 py-1 rounded text-sm ${
          destroyEnabled
            ? "bg-db-red-500 hover:bg-db-red-600 text-white"
            : "bg-db-red-300 text-db-red-100 cursor-default"
        }`}
        onClick={() => destroyMutation.mutate(universe)}
        disabled={!destroyEnabled}
      >
        Löschen
      </button>
      <button
        type="button"
        className={`px-3 py-1 rounded text-sm ${
          requestFromServerEnabled
            ? "bg-db-red-500 hover:bg-db-red-600 text-white"
            : "bg-db-red-300 text-db-red-100 cursor-default"
        }`}
        onClick={() => requestFromServerMutation.mutate()}
        disabled={!requestFromServerEnabled}
        title="Liste der Paralleluniversen vom Server laden"
      >
        <ArrowDownIcon className="h-5 w-5" />
      </button>
      {universes.map((uv) => (
        <button
          key={uv.id}
          type="button"
          className={`px-3 py-1 rounded text-white text-sm ${
            isMutating
              ? uv.id == universe
                ? "bg-db-red-300 text-db-red-100 ring ring-db-red-800 cursor-default"
                : "bg-db-red-300 text-db-red-100 cursor-default"
              : uv.id == universe
              ? "bg-db-red-500 ring ring-db-red-800"
              : "bg-db-red-500 hover:bg-db-red-600"
          }`}
          onClick={() => switchTo(uv)}
          disabled={isMutating}
          title={`Universum ${uv.id} (Fahrplan ${uv.schedule})`}
        >
          #{uv.id}
        </button>
      ))}
    </div>
  );
}

export default UniverseControl;
