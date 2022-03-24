import { AxisBottom } from "@visx/axis";
import { GridColumns } from "@visx/grid";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { ViolinPlot } from "@visx/stats";

import { PaxMonPdfEntry } from "@/api/protocol/motis/paxmon";

import { PaxMonEdgeLoadInfoWithStats } from "@/data/loadInfo";

const defaultMargin = {
  top: 0,
  right: 10,
  bottom: 25,
  left: 10,
};

type SectionLoadGraphProps = {
  section: PaxMonEdgeLoadInfoWithStats;
  width: number;
  height: number;
  maxVal?: number | undefined;
  margin?: typeof defaultMargin;
};

function SectionLoadGraph({
  section,
  width,
  height,
  maxVal,
  margin = defaultMargin,
}: SectionLoadGraphProps): JSX.Element {
  margin ??= defaultMargin;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const count = (e: PaxMonPdfEntry) => e.p;
  const value = (e: PaxMonPdfEntry) => e.n;

  const paxLimit = maxVal ?? section.max_pax;
  const paxScale = scaleLinear<number>({
    domain: [0, paxLimit],
    range: [margin.left, margin.left + innerWidth],
    round: true,
  });

  const bgSections = [];
  if (section.capacity > 0) {
    const y = margin.top;
    const h = innerHeight;
    const x0 = paxScale(0);
    const xMax = margin.left + innerWidth;
    const x80 = Math.min(xMax, paxScale(section.capacity * 0.8));
    const x100 = Math.min(xMax, paxScale(section.capacity));
    const x120 = Math.min(xMax, paxScale(section.capacity * 1.2));
    const x200 = Math.min(xMax, paxScale(section.capacity * 2.0));
    bgSections.push(
      <rect
        key="0-80"
        x={x0}
        y={y}
        width={x80 - x0}
        height={h}
        fill="#C9EB9E"
      />,
      <rect
        key="80-100"
        x={x80}
        y={y}
        width={x100 - x80}
        height={h}
        fill="#FFFFAF"
      />,
      <rect
        key="100-120"
        x={x100}
        y={y}
        width={x120 - x100}
        height={h}
        fill="#FCE3B4"
      />,
      <rect
        key="120-200"
        x={x120}
        y={y}
        width={x200 - x120}
        height={h}
        fill="#FCC8C3"
      />
    );
    if (x200 < width) {
      bgSections.push(
        <rect
          key="200+"
          x={x200}
          y={y}
          width={Math.max(0, margin.left + innerWidth - x200)}
          height={h}
          fill="#FA9090"
        />
      );
    }
  } else {
    bgSections.push(
      <rect
        key="unknown"
        x={margin.left}
        y={margin.top}
        width={innerWidth}
        height={innerHeight}
        fill="white"
      />
    );
  }

  return (
    <svg width={width} height={height}>
      <g>{bgSections}</g>
      <GridColumns
        scale={paxScale}
        top={margin.top}
        height={innerHeight}
        stroke="#eee"
        strokeOpacity={0.5}
        numTicks={paxLimit / 10}
      />
      <ViolinPlot
        data={section.dist.pdf}
        stroke="#3038FF"
        strokeWidth={2}
        fill="#B2B5FE"
        valueScale={paxScale}
        count={count}
        value={value}
        top={margin.top + 4}
        width={innerHeight - 8}
        horizontal={true}
      />
      <path
        d={`M${paxScale(section.expected_passengers)} ${
          margin.top
        } v${innerHeight}`}
        stroke="#333"
        strokeDasharray={2}
        strokeWidth={2}
      />
      <AxisBottom scale={paxScale} top={margin.top + innerHeight} />
    </svg>
  );
}

function ResponsiveSectionLoadGraph(
  props: Omit<SectionLoadGraphProps, "width" | "height">
): JSX.Element {
  return (
    <ParentSize>
      {({ width, height }) => (
        <SectionLoadGraph width={width} height={height} {...props} />
      )}
    </ParentSize>
  );
}

export default ResponsiveSectionLoadGraph;
