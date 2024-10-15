import { useAtom } from "jotai";
import { useState } from "react";
import { useIsMutating, useMutation } from "react-query";

import {
  sendPaxMonDestroyUniverseRequest,
  sendPaxMonForkUniverseRequest,
} from "@/api/paxmon";

import { scheduleAtom, universeAtom } from "@/data/simulation";

type Universe = {
  universe: number;
  schedule: number;
};

function UniverseControl(): JSX.Element {
  const [universe, setUniverse] = useAtom(universeAtom);
  const [schedule, setSchedule] = useAtom(scheduleAtom);
  const [universes, setUniverses] = useState<Universe[]>([
    { universe: 0, schedule: 0 },
  ]);
  const isMutating = useIsMutating() != 0;

  function switchTo(uv: Universe) {
    setUniverse(uv.universe);
    setSchedule(uv.schedule);
  }

  const forkMutation = useMutation(
    (baseUniverse: number) =>
      sendPaxMonForkUniverseRequest({
        universe: baseUniverse,
        fork_schedule: true,
      }),
    {
      onSuccess: (data) => {
        const newUv = { universe: data.universe, schedule: data.schedule };
        setUniverses([
          ...universes.filter((uv) => uv.universe !== data.universe),
          newUv,
        ]);
        switchTo(newUv);
      },
    }
  );
  const destroyMutation = useMutation(
    (uv: number) => sendPaxMonDestroyUniverseRequest({ universe: uv }),
    {
      onSettled: (data, error, variables) => {
        if (error) {
          console.log(
            `error while trying to destroy universe ${variables}:`,
            error
          );
        }
        setUniverses(universes.filter((u) => u.universe != variables));
        switchTo(universes[0]);
      },
    }
  );

  const forkEnabled = !isMutating;
  const destroyEnabled = !isMutating && universe != 0;

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
      {universes.map((uv) => (
        <button
          key={uv.universe}
          type="button"
          className={`px-3 py-1 rounded text-white text-sm ${
            isMutating
              ? uv.universe == universe
                ? "bg-db-red-300 text-db-red-100 ring ring-db-red-800 cursor-default"
                : "bg-db-red-300 text-db-red-100 cursor-default"
              : uv.universe == universe
              ? "bg-db-red-500 ring ring-db-red-800"
              : "bg-db-red-500 hover:bg-db-red-600"
          }`}
          onClick={() => switchTo(uv)}
          disabled={isMutating}
          title={`Universum ${uv.universe} (Fahrplan ${uv.schedule})`}
        >
          #{uv.universe}
        </button>
      ))}
    </div>
  );
}

export default UniverseControl;
