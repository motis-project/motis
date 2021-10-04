import React, { useState } from "react";
import { QueryClient, useMutation, useQueryClient } from "react-query";
import { useAtom } from "jotai";

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

  return (
    <div className="flex justify-center items-baseline space-x-2 pl-2">
      <span>Universum #{universe}</span>
      {universes.map((uv) => (
        <button
          key={uv}
          type="button"
          className={`px-3 py-1 rounded-xl text-white text-sm ${
            uv == universe ? "bg-blue-500" : "bg-blue-900 hover:bg-blue-800"
          }`}
          onClick={() => setUniverse(uv)}
        >
          #{uv}
        </button>
      ))}
      <button
        type="button"
        className={`bg-blue-900 px-3 py-1 rounded-xl text-white text-sm ${
          forkEnabled ? "hover:bg-blue-800" : "text-blue-700"
        }`}
        onClick={() => forkMutation.mutate(universe)}
        disabled={!forkEnabled}
      >
        Neu
      </button>
      <button
        type="button"
        className={`bg-blue-900 px-3 py-1 rounded-xl text-white text-sm ${
          destroyEnabled ? "hover:bg-blue-800" : "text-blue-700"
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
