import { RectClipPath } from "@visx/clip-path";
import { localPoint } from "@visx/event";
import { Group } from "@visx/group";
import { Tree, hierarchy } from "@visx/hierarchy";
import { HierarchyPointNode } from "@visx/hierarchy/lib/types";
import { ParentSize } from "@visx/responsive";
import { LinkVertical } from "@visx/shape";
import { Zoom } from "@visx/zoom";
import { TransformMatrix } from "@visx/zoom/lib/types";
import { Download, Maximize } from "lucide-react";
import { ReactNode, useMemo } from "react";

import {
  PaxMonGroup,
  PaxMonLocalizationWrapper,
  PaxMonRerouteLogEntry,
  PaxMonRerouteReason,
} from "@/api/protocol/motis/paxmon";

import { formatPercent } from "@/data/numberFormat";

import { saveAsSVG } from "@/util/download";
import { canSwitchLocalization } from "@/util/paxLocalization";

import { Button } from "@/components/ui/button";

interface TreeNode {
  children?: TreeNode[];
  color: string;

  route: number;
  pickProbability: number;
  probability: number;
  version: number;
}

type HierarchyNode = HierarchyPointNode<TreeNode>;

interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

const nodeWidth = 50;
const nodeHeight = 50;
const horizontalSpacing = 10;
const verticalSpacing = 30;

const defaultMargin: Margin = { top: 40, left: 10, right: 10, bottom: 40 };

const NodeColors = {
  Active: "#86efac", // green-300
  Manual: "#93c5fd", // blue-300
  BrokenTransfer: "#fca5a5", // red-300
  MajorDelayExpected: "#fcd34d", // amber-300
  RevertForecast: "#5eead4", // teal-300
  Simulation: "#67e8f9", // cyan-300
  UpdateForecast: "#c4b5fd", // violet-300
  DestinationUnreachable: "#f0abfc", // fuchsia-300
  DestinationReachable: "#bef264", // lime-300
};

function setNodeColor(node: TreeNode, reason: PaxMonRerouteReason) {
  switch (reason) {
    case "Manual":
      node.color = NodeColors.Manual;
      break;
    case "BrokenTransfer":
      node.color = NodeColors.BrokenTransfer;
      break;
    case "MajorDelayExpected":
      node.color = NodeColors.MajorDelayExpected;
      break;
    case "RevertForecast":
      node.color = NodeColors.RevertForecast;
      break;
    case "Simulation":
      node.color = NodeColors.Simulation;
      break;
    case "UpdateForecast":
      node.color = NodeColors.UpdateForecast;
      break;
    case "DestinationUnreachable":
      node.color = NodeColors.DestinationUnreachable;
      break;
    case "DestinationReachable":
      node.color = NodeColors.DestinationReachable;
      break;
  }
}

interface PaxMonRerouteLogEntryWithVersion {
  le: PaxMonRerouteLogEntry;
  version: number;
}

export function buildRouteTree(group: PaxMonGroup): TreeNode {
  const root: TreeNode = {
    color: NodeColors.Active,
    pickProbability: 1.0,
    probability: 1.0,
    route: 0,
    version: 0,
  };

  const leaves: TreeNode[][] = Array.from(
    { length: group.routes.length },
    () => [],
  );

  leaves[0].push(root);

  const parents = new Map<TreeNode, TreeNode>();
  let reverts: PaxMonRerouteLogEntryWithVersion[] = [];

  const processReverts = () => {
    if (reverts.length === 0) {
      return;
    }
    const reactivatedRoutes = reverts.map((lev) => lev.le.old_route.index);
    const revertedRoutes = new Set<number>();
    const reactivatedRouteLogEntry = new Map<
      number /* route index */,
      PaxMonRerouteLogEntryWithVersion
    >();
    const localizations = new Map<
      number /* route index */,
      PaxMonLocalizationWrapper
    >();

    for (const lev of reverts) {
      reactivatedRouteLogEntry.set(lev.le.old_route.index, lev);
      localizations.set(lev.le.old_route.index, lev.le.old_route);
      for (const nr of lev.le.new_routes) {
        localizations.set(nr.index, nr);
        if (nr.previous_probability > nr.new_probability) {
          revertedRoutes.add(nr.index);
        }
      }
    }

    for (const revertedRoute of revertedRoutes) {
      const leafLocalization = localizations.get(revertedRoute);
      if (!leafLocalization) {
        throw new Error(
          `leaf localization not found for route ${revertedRoute}`,
        );
      }
      const oldLeaves = leaves[revertedRoute];
      leaves[revertedRoute] = [];
      for (const oldLeaf of oldLeaves) {
        let candidate: TreeNode | null = null;
        for (
          let node = parents.get(oldLeaf);
          node !== undefined;
          node = parents.get(node)
        ) {
          if (reactivatedRoutes.includes(node.route)) {
            const candidateLocalization = localizations.get(node.route);
            if (!candidateLocalization) {
              throw new Error(
                `candidate localization not found for route ${candidateLocalization}`,
              );
            }
            if (
              canSwitchLocalization(leafLocalization, candidateLocalization)
            ) {
              candidate = node;
            }
          }
        }
        if (candidate) {
          const lev = reactivatedRouteLogEntry.get(candidate.route);
          if (!lev) {
            throw new Error("no reactivated route log entry found");
          }
          setNodeColor(oldLeaf, lev.le.reason);
          const newNode: TreeNode = {
            color: NodeColors.Active,
            pickProbability: 1,
            probability: oldLeaf.probability,
            route: candidate.route,
            version: lev.version + 1,
          };
          oldLeaf.children ||= [];
          oldLeaf.children.push(newNode);
          leaves[newNode.route].push(newNode);
          parents.set(newNode, oldLeaf);
        } else {
          leaves[revertedRoute].push(oldLeaf);
        }
      }
    }

    reverts = [];
  };

  for (const [version, le] of group.reroute_log.entries()) {
    if (le.reason === "RevertForecast") {
      if (
        reverts.length > 0 &&
        reverts[0].le.update_number !== le.update_number
      ) {
        processReverts();
      }
      reverts.push({ le, version });
    } else {
      processReverts();
      const oldRoute = le.old_route.index;
      const prevNodes = leaves[oldRoute];
      if (prevNodes.length === 0 && le.old_route.previous_probability == 0) {
        continue;
      }
      console.assert(
        prevNodes.length !== 0,
        "old nodes not found (version %i): %o",
        version + 1,
        le,
      );
      leaves[oldRoute] = [];

      let totalOutgoingProbability = 0;
      for (const newRoute of le.new_routes) {
        if (newRoute.index === oldRoute) {
          totalOutgoingProbability += newRoute.new_probability;
        } else {
          totalOutgoingProbability +=
            newRoute.new_probability - newRoute.previous_probability;
        }
      }

      for (const prevNode of prevNodes) {
        setNodeColor(prevNode, le.reason);

        for (const newRoute of le.new_routes) {
          const probChange =
            newRoute.index === oldRoute
              ? newRoute.new_probability
              : newRoute.new_probability - newRoute.previous_probability;
          const pickProbability = probChange / totalOutgoingProbability;
          const newNode: TreeNode = {
            color: NodeColors.Active,
            pickProbability,
            probability: pickProbability * prevNode.probability,
            route: newRoute.index,
            version: version + 1,
          };
          prevNode.children ||= [];
          prevNode.children.push(newNode);
          leaves[newNode.route].push(newNode);
          parents.set(newNode, prevNode);
        }
      }
    }
  }
  processReverts();

  return root;
}

function Node({ node }: { node: HierarchyNode }) {
  const centerX = -nodeWidth / 2;
  const centerY = -nodeHeight / 2;

  return (
    <Group top={node.y} left={node.x}>
      <rect
        height={nodeHeight}
        width={nodeWidth}
        y={centerY}
        x={centerX}
        fill={node.data.color}
        stroke={"#444"}
        rx={10}
        onClick={() => {
          alert(`clicked: ${JSON.stringify(node.data, null, 2)}`);
        }}
      />
      <text
        dy="-0.5em"
        fontSize={12}
        fontFamily="Arial"
        textAnchor="middle"
        fill={"#000"}
        style={{ pointerEvents: "none" }}
      >
        #{node.data.route}
      </text>
      <text
        dy="0.6em"
        fontSize={9}
        fontFamily="Arial"
        textAnchor="middle"
        fill={"#222"}
        style={{ pointerEvents: "none" }}
      >
        {formatPercent(node.data.probability, { maximumFractionDigits: 0 })}
      </text>
      <text
        dy="1.8em"
        fontSize={9}
        fontFamily="Arial"
        textAnchor="middle"
        fill={"#555"}
        style={{ pointerEvents: "none" }}
      >
        V{node.data.version}
      </text>
    </Group>
  );
}

function getMinMaxX(root: HierarchyNode): [number, number] {
  let min = Infinity;
  let max = -Infinity;

  root.each((node) => {
    min = Math.min(min, node.x);
    max = Math.max(max, node.x);
  });

  min -= nodeWidth / 2;
  max += nodeWidth / 2;

  return [min, max];
}

function getInitialTransform(
  root: HierarchyNode,
  width: number,
  height: number,
  totalHeight: number,
): TransformMatrix {
  // the root node is always at (0, 0)
  // determine the initial scale so that the whole tree is visible
  // TODO: check height as well
  const [minX, maxX] = getMinMaxX(root);
  const totalWidth = maxX - minX;
  const scale = Math.min(1.0, width / totalWidth);

  // if the whole tree fits at scale 1, center it in the viewport
  // otherwise, make it fit the viewport
  const translateX =
    totalWidth < width
      ? (-minX + (width - totalWidth) / 2) * scale
      : -minX * scale;
  const scaledHeight = totalHeight * scale;
  const translateY = scaledHeight >= height ? 0 : (height - scaledHeight) / 2;

  // console.log(
  //   `getInitialTransform: width=${width}, totalWidth=${totalWidth}, minX=${minX}, maxX=${maxX}, scale=${scale}, translateX=${translateX}`,
  // );

  return {
    scaleX: scale,
    scaleY: scale,
    skewX: 0,
    skewY: 0,
    translateX: translateX,
    translateY: translateY,
  };
}

export interface GroupRouteTreeProps {
  group: PaxMonGroup;
  width: number;
  height?: number | undefined;
  margin?: Margin;
}

function GroupRouteTree({
  group,
  width,
  height,
  margin = defaultMargin,
}: GroupRouteTreeProps) {
  const data = useMemo(() => hierarchy(buildRouteTree(group)), [group]);

  const totalHeight =
    margin.top + margin.bottom + data.height * (nodeHeight + verticalSpacing);
  const elementHeight = height ?? totalHeight;

  const yMax = elementHeight - margin.top - margin.bottom;
  const xMax = width - margin.left - margin.right;

  return width < 10 ? null : (
    <Tree<TreeNode>
      root={data}
      size={[xMax, yMax]}
      nodeSize={[nodeWidth + horizontalSpacing, nodeHeight + verticalSpacing]}
    >
      {(tree) => (
        <Zoom<SVGSVGElement>
          width={width}
          height={elementHeight}
          scaleXMax={4}
          scaleYMax={4}
          initialTransformMatrix={getInitialTransform(
            tree,
            xMax,
            elementHeight,
            totalHeight,
          )}
        >
          {(zoom) => (
            <div className="relative">
              <svg
                width={width}
                height={elementHeight}
                style={{
                  cursor: zoom.isDragging ? "grabbing" : "grab",
                  touchAction: "none",
                }}
                ref={zoom.containerRef}
              >
                <RectClipPath
                  id="zoom-clip"
                  width={width}
                  height={elementHeight}
                />
                <rect
                  width={width}
                  height={elementHeight}
                  rx={14}
                  fill={"#fff"}
                />
                <g transform={zoom.toString()}>
                  <Group top={margin.top} left={margin.left}>
                    {tree.links().map((link, i) => (
                      <LinkVertical
                        key={`link-${i}`}
                        data={link}
                        stroke={"#000"}
                        strokeWidth="1"
                        fill="none"
                      />
                    ))}
                    {tree.descendants().map((node, i) => (
                      <Node key={`node-${i}`} node={node} />
                    ))}
                  </Group>
                </g>
                <rect
                  width={width}
                  height={elementHeight}
                  rx={14}
                  fill="transparent"
                  onTouchStart={zoom.dragStart}
                  onTouchMove={zoom.dragMove}
                  onTouchEnd={zoom.dragEnd}
                  onMouseDown={zoom.dragStart}
                  onMouseMove={zoom.dragMove}
                  onMouseUp={zoom.dragEnd}
                  onMouseLeave={() => {
                    if (zoom.isDragging) zoom.dragEnd();
                  }}
                  onDoubleClick={(event) => {
                    const point = localPoint(event) ?? { x: 0, y: 0 };
                    zoom.scale({ scaleX: 1.1, scaleY: 1.1, point });
                  }}
                />
              </svg>
              <div className="absolute bottom-2 right-2 flex flex-col gap-2">
                <Button
                  variant="outline"
                  onClick={() =>
                    saveAsSVG(zoom.containerRef.current, `group-${group.id}`)
                  }
                >
                  <Download className="h-4 w-4" />
                </Button>
                <Button variant="outline" onClick={zoom.reset}>
                  <Maximize className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </Zoom>
      )}
    </Tree>
  );
}

export type ResponsiveGroupRouteTreeProps = Omit<GroupRouteTreeProps, "width">;

function ResponsiveGroupRouteTree(
  props: ResponsiveGroupRouteTreeProps,
): ReactNode {
  return (
    <ParentSize>
      {({ width }) => <GroupRouteTree width={width} {...props} />}
    </ParentSize>
  );
}

export default ResponsiveGroupRouteTree;
