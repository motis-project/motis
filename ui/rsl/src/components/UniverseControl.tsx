import React, { useState } from "react";
import { useMutation } from "react-query";
import { useAtom } from "jotai";
import { PlusCircleIcon, XCircleIcon } from "@heroicons/react/solid";

import { universeAtom } from "../data/simulation";
import {
  sendPaxMonDestroyUniverseRequest,
  sendPaxMonForkUniverseRequest,
} from "../api/paxmon";

function UniverseControl(): JSX.Element {
  const [universe, setUniverse] = useAtom(universeAtom);
  const [universes, setUniverses] = useState([0]);

  const forkMutation = useMutation(
    (baseUniverse: number) =>
      sendPaxMonForkUniverseRequest({ universe: baseUniverse }),
    {
      onSuccess: (data) => {
        setUniverses([...universes, data.universe]);
        setUniverse(data.universe);
      },
    }
  );
  const destroyMutation = useMutation(
    (uv: number) => sendPaxMonDestroyUniverseRequest({ universe: uv }),
    {
      onSuccess: (data, variables) => {
        setUniverses(universes.filter((u) => u != variables));
        setUniverse(0);
      },
    }
  );

  const forkEnabled = !forkMutation.isLoading && !destroyMutation.isLoading;
  const destroyEnabled = forkEnabled && universe != 0;

  // <PlusCircleIcon className="h-5 w-5 text-white" />
  // <XCircleIcon className="h-5 w-5 text-white" />

  return (
    <div className="flex justify-center items-center space-x-2 pl-4">
      <span className="pr-2">Universum #{universe}</span>
      {universes.map((uv) => (
        <button
          key={uv}
          type="button"
          className={`px-3 py-1 rounded text-white text-sm ${
            uv == universe
              ? "bg-db-red-500 ring ring-db-red-800"
              : "bg-db-red-500 hover:bg-db-red-600"
          }`}
          onClick={() => setUniverse(uv)}
        >
          #{uv}
        </button>
      ))}
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
        Neu
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
    </div>
  );
}

export default UniverseControl;
