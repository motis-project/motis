import { useAtom } from "jotai";
import { useQuery, useQueryClient } from "react-query";

import { TripId } from "@/api/protocol/motis";
import { PaxMonEdgeLoadInfo } from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonGetTripLoadInfosRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/simulation";

export type MiniTripLoadGraphProps = {
  edges: PaxMonEdgeLoadInfo[];
};

interface SectionGeometry {
  x: number;
  width: number;
  forecastHeight: number;
  forecastColor: string;
  capacityHeight: number;
  expectedHeight: number;
}

function getSectionColor(capacity: number, maxLoad: number) {
  if (capacity == 0) {
    return "#3C414B";
  } else {
    const load = maxLoad / capacity;
    if (load > 2.0) {
      return "#C50014";
    } else if (load > 1.2) {
      return "#EC0016";
    } else if (load > 1.0) {
      return "#F39200";
    } else if (load > 0.8) {
      return "#FFD800";
    } else {
      return "#408335";
    }
  }
}

function MiniTripLoadGraph({ edges }: MiniTripLoadGraphProps): JSX.Element {
  const graphWidth = 1000;
  const graphHeight = 100;

  const maxEdgeLoad = (eli: PaxMonEdgeLoadInfo) => eli.dist.q95;

  const maxCapacity = edges.reduce(
    (max, eli) => (eli.capacity ? Math.max(max, eli.capacity) : max),
    0
  );
  const totalMaxPax = edges.reduce(
    (max, eli) => Math.max(max, maxEdgeLoad(eli)),
    0
  );
  const totalMaxLoad = Math.max(maxCapacity, totalMaxPax, 1);

  const loadHeight = (load: number) => (load / totalMaxLoad) * graphHeight;

  const sectionDurations = edges.map((e) =>
    Math.max(300, e.arrival_schedule_time - e.departure_schedule_time)
  );
  const totalDuration = sectionDurations.reduce((sum, v) => sum + v);
  const sectionGeometry = edges.reduce(
    (a, eli, idx) => {
      const capacity = eli.capacity;
      const maxLoad = maxEdgeLoad(eli);
      const sg = {
        x: a.x,
        width: (sectionDurations[idx] / totalDuration) * graphWidth,
        forecastHeight: loadHeight(maxLoad),
        forecastColor: getSectionColor(capacity, maxLoad),
        capacityHeight: loadHeight(capacity),
        expectedHeight: loadHeight(eli.expected_passengers),
      };
      return { x: sg.x + sg.width, sections: [...a.sections, sg] };
    },
    { x: 0, sections: [] as SectionGeometry[] }
  ).sections;

  return (
    <svg viewBox={`0 0 ${graphWidth} ${graphHeight}`} className="w-full">
      <g>
        {sectionGeometry.map((sg, idx) => (
          <g key={idx}>
            <rect
              x={sg.x}
              y={graphHeight - sg.forecastHeight}
              width={sg.width}
              height={sg.forecastHeight}
              fill={sg.forecastColor}
            />
          </g>
        ))}
      </g>
    </svg>
  );
}

export default MiniTripLoadGraph;

export type MiniTripLoadGraphForTripProps = {
  tripId: TripId;
};

export function MiniTripLoadGraphForTrip({
  tripId,
}: MiniTripLoadGraphForTripProps): JSX.Element | null {
  const [universe] = useAtom(universeAtom);

  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery(
    queryKeys.tripLoad(universe, tripId),
    () => sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    {
      placeholderData: () => {
        return universe != 0
          ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
          : undefined;
      },
    }
  );

  return data ? <MiniTripLoadGraph edges={data.load_infos[0].edges} /> : null;
}
