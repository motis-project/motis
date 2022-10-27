import * as d3Force from "d3-force";
import { SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import { useEffect, useState } from "react";

import {
  PaxMonDebugEdge,
  PaxMonDebugGraphResponse,
  PaxMonDebugNode,
} from "@/api/protocol/motis/paxmon";

type SimpleGraphProps = {
  data: PaxMonDebugGraphResponse;
  width: number;
  height: number;
};

interface Node extends SimulationNodeDatum {
  id: number;
}

type Edge = SimulationLinkDatum<Node>;

function SimpleGraph({ data, width, height }: SimpleGraphProps) {
  const [d3Nodes, setD3Nodes] = useState<Node[]>([]);
  const [d3Edges, setD3Edges] = useState<Edge[]>([]);

  useEffect(() => {
    setD3Nodes([]);
    setD3Edges([]);
    if (!data) {
      return;
    }

    let minTime = Number.MAX_VALUE;
    let maxTime = 1;
    for (const n of data.nodes) {
      if (n.schedule_time != 0) {
        minTime = Math.min(minTime, n.schedule_time);
        maxTime = Math.max(maxTime, n.schedule_time);
      }
    }
    const timeRange = maxTime - minTime;

    const nodes: Node[] = data.nodes.map((n) => {
      return {
        id: n.index,
        x:
          n.schedule_time != 0
            ? 100 + ((n.schedule_time - minTime) / timeRange) * (width - 200)
            : 0,
        y: height / 2,
      };
    });

    const links: Edge[] = data.edges.map((e, i) => {
      return { source: e.from_node, target: e.to_node, index: i };
    });

    console.log("graph init node pos:", nodes.map((n) => n.x).join(", "));

    const sim = d3Force
      .forceSimulation<Node>(nodes)
      .force("center", d3Force.forceCenter(width / 2, height / 2))
      .force(
        "repel",
        d3Force.forceManyBody().strength(-30).distanceMin(10).distanceMax(150)
      )
      .force("collision", d3Force.forceCollide().radius(10))
      .force(
        "link",
        d3Force
          .forceLink<Node, Edge>()
          .id((n) => n.id)
          .links(links)
      )
      .on("tick", () => {
        setD3Nodes([...nodes]);
        setD3Edges(links);
      })
      .on("end", () => {
        console.log("SimpleGraph: animation done");
        console.log("nodes:", nodes);
        console.log("links:", links);
      });

    return () => {
      console.log("SimpleGraph: useEffect clean-up");
      sim.stop();
    };
  }, [data, width, height]);

  if (
    !data ||
    data.nodes.length !== d3Nodes.length ||
    data.edges.length !== d3Edges.length
  ) {
    return <></>;
  }

  return (
    <div>
      <svg width={width} height={height}>
        <g>
          {d3Edges.map((e) => (
            <GraphEdge
              key={e.index}
              d3Edge={e}
              paxmonEdge={data.edges[e.index || 0]}
            />
          ))}
        </g>
        <g>
          {d3Nodes.map((n) => (
            <GraphNode
              key={n.index}
              d3Node={n}
              paxmonNode={data.nodes[n.index || 0]}
            />
          ))}
        </g>
      </svg>
    </div>
  );
}

type GraphNodeProps = {
  d3Node: Node;
  paxmonNode: PaxMonDebugNode;
};

function GraphNode({ d3Node, paxmonNode }: GraphNodeProps): JSX.Element {
  return (
    <circle
      key={d3Node.index}
      cx={d3Node.x}
      cy={d3Node.y}
      r={5}
      stroke="black"
      fill={getNodeColor(paxmonNode)}
    />
  );
}

function getNodeColor(node: PaxMonDebugNode): string {
  if (node.valid) {
    return "white";
  } else {
    return "red";
  }
}

type GraphEdgeProps = {
  d3Edge: Edge;
  paxmonEdge: PaxMonDebugEdge;
};

function GraphEdge({ d3Edge, paxmonEdge }: GraphEdgeProps): JSX.Element {
  const source = d3Edge.source as Node;
  const target = d3Edge.target as Node;
  return (
    <line
      x1={source.x}
      y1={source.y}
      x2={target.x}
      y2={target.y}
      stroke={getEdgeColor(paxmonEdge)}
    />
  );
}

function getEdgeColor(edge: PaxMonDebugEdge): string {
  switch (edge.type) {
    case "Trip":
      return "black";
    case "Interchange":
      return edge.broken ? "red" : "lime";
    case "Wait":
      return "blue";
    case "Through":
      return "orange";
    case "Disabled":
      return "#aaa";
  }
}

export default SimpleGraph;
