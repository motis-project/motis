import { PaxMonEdgeLoadInfo } from "@/api/protocol/motis/paxmon";

export type LegacyMiniTripLoadGraphProps = {
  edges: PaxMonEdgeLoadInfo[];
};

function LegacyMiniTripLoadGraph({
  edges,
}: LegacyMiniTripLoadGraphProps): JSX.Element {
  const graphWidth = 1000;
  const graphHeight = 100;
  const minEdgeLoad = (eli: PaxMonEdgeLoadInfo) => eli.dist.min;
  const maxEdgeLoad = (eli: PaxMonEdgeLoadInfo) => eli.dist.max;
  const avgEdgeLoad = (eli: PaxMonEdgeLoadInfo) => eli.dist.q50;

  const maxCapacity = edges.reduce(
    (max, eli) => (eli.capacity ? Math.max(max, eli.capacity) : max),
    0
  );
  const maxPax = edges.reduce((max, eli) => Math.max(max, maxEdgeLoad(eli)), 0);
  const maxLoad = Math.max(maxCapacity, maxPax, 1);

  const sectionWidth = graphWidth / edges.length;
  const sectionGeometry = edges.reduce(
    (a, e, idx) => [...a, { x: idx * sectionWidth, width: sectionWidth }],
    [] as Array<{ x: number; width: number }>
  );

  const loadY = (load: number) => (1 - load / maxLoad) * graphHeight;

  // TODO:
  // - diff graph - 2 edge sets
  // - if trip route was changed, show new route and only matching edges
  // - show +/- pax per section (min/max/avg?)
  // - section infos (station names) @hover?

  return (
    <svg viewBox={`0 0 ${graphWidth} ${graphHeight}`} className="w-full">
      <g>
        {edges.map((eli, idx) => {
          const capacity = eli.capacity;
          const { x: left, width } = sectionGeometry[idx];
          const bgs = [];
          if (capacity) {
            const yRed = 0;
            const yYellow = loadY(capacity);
            const yGreen = loadY(capacity * 0.8);
            const hRed = yYellow - yRed;
            const hYellow = yGreen - yYellow;
            const hGreen = graphHeight - yGreen;

            if (hRed > 0) {
              bgs.push(
                <rect
                  x={left}
                  y={yRed}
                  width={width}
                  height={hRed}
                  fill="#FFCACA"
                  key="red"
                />
              );
            }
            if (hYellow > 0) {
              bgs.push(
                <rect
                  x={left}
                  y={yYellow}
                  width={width}
                  height={hYellow}
                  fill="#FFF3CA"
                  key="yellow"
                />
              );
            }
            if (hGreen > 0) {
              bgs.push(
                <rect
                  x={left}
                  y={yGreen}
                  width={width}
                  height={hGreen}
                  fill="#D4FFCA"
                  key="green"
                />
              );
            }
          } else {
            bgs.push(
              <rect
                x={left}
                y="0"
                width={width}
                height={graphHeight}
                fill="white"
                key="white"
              />
            );
          }
          return <g key={idx}>{bgs}</g>;
        })}
      </g>
      <g>
        {edges.map((eli, idx) => (
          <path
            key={idx}
            className="fill-indigo-500 opacity-40"
            d={`M ${sectionGeometry[idx].x} ${loadY(maxEdgeLoad(eli))} h ${
              sectionGeometry[idx].width
            } V ${loadY(minEdgeLoad(eli))} h ${-sectionGeometry[idx].width} Z`}
          />
        ))}
      </g>
      <g>
        {edges.map((eli, idx) => (
          <path
            key={idx}
            className="stroke-indigo-800 stroke-2"
            d={`M ${sectionGeometry[idx].x} ${loadY(avgEdgeLoad(eli))} h ${
              sectionGeometry[idx].width
            }`}
          />
        ))}
      </g>
      <g>
        {edges.map((_, idx) => (
          <path
            key={idx}
            stroke="#aaa"
            d={`M ${sectionGeometry[idx].x} 0 V ${graphHeight}`}
          />
        ))}
      </g>
      <rect
        x="0"
        y="0"
        width={graphWidth}
        height={graphHeight}
        fill="transparent"
        stroke="#333"
      />
    </svg>
  );
}

export default LegacyMiniTripLoadGraph;
