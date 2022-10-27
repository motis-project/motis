import { useQuery } from "@tanstack/react-query";
import { useAtom } from "jotai";
import { useState } from "react";

import {
  PaxMonDebugGraphRequest,
  PaxMonGroup,
} from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonDebugGraphRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import SimpleGraph from "@/components/groups/SimpleGraph";

type DebugGraphProps = {
  group: PaxMonGroup;
};

function DebugGraph({ group }: DebugGraphProps): JSX.Element {
  const [selectedRoutes, setSelectedRoutes] = useState<Array<number>>([]);

  const isRouteSelected = (i: number) => selectedRoutes.includes(i);
  const toggleRouteSelected = (i: number) => {
    if (isRouteSelected(i)) {
      setSelectedRoutes((sr) => sr.filter((v) => v !== i));
    } else {
      setSelectedRoutes((sr) => [...sr, i].sort());
    }
  };

  const [universe] = useAtom(universeAtom);
  const req: PaxMonDebugGraphRequest = {
    universe: universe,
    node_indices: [],
    group_routes: selectedRoutes.map((r) => {
      return { g: group.id, r };
    }),
    trip_ids: [],
    filter_groups: true,
    include_full_trips_from_group_routes: true,
    include_canceled_trip_nodes: true,
  };
  const { data /*, isLoading, error */ } = useQuery(
    queryKeys.debugGraph(req),
    () => sendPaxMonDebugGraphRequest(req),
    { enabled: selectedRoutes.length > 0 }
  );

  return (
    <div className="space-y-2">
      <p className="text-xl">Debug-Graph</p>
      <div className="flex flex-wrap gap-2">
        <span>Routen:</span>
        {group.routes.map((route) => (
          <label key={route.index}>
            <input
              type="checkbox"
              checked={isRouteSelected(route.index)}
              onChange={() => toggleRouteSelected(route.index)}
            />
            #{route.index}
          </label>
        ))}
      </div>
      {data && <SimpleGraph data={data} width={1500} height={1000} />}
    </div>
  );
}

export default DebugGraph;
