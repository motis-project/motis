import { AxisBottom } from "@visx/axis";
import { GridColumns } from "@visx/grid";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { BoxPlot, ViolinPlot } from "@visx/stats";

import {
  PaxMonEdgeLoadInfo,
  PaxMonPdfEntry,
} from "@/api/protocol/motis/paxmon";

import { SectionLoadColors } from "@/util/colors";

export type SectionLoadGraphPlotType = "SimpleBox" | "Violin" | "Box";

const defaultMargin = {
  top: 0,
  right: 10,
  bottom: 25,
  left: 10,
};

type SectionLoadGraphProps = {
  section: PaxMonEdgeLoadInfo;
  width: number;
  height: number;
  maxVal?: number | undefined;
  margin?: typeof defaultMargin;
  plotType?: SectionLoadGraphPlotType;
};

function SectionLoadGraph({
  section,
  width,
  height,
  maxVal,
  margin = defaultMargin,
  plotType = "SimpleBox",
}: SectionLoadGraphProps): JSX.Element {
  margin ??= defaultMargin;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  if (innerWidth < 0 || innerHeight < 0) {
    return <></>;
  }

  const count = (e: PaxMonPdfEntry) => e.p;
  const value = (e: PaxMonPdfEntry) => e.n;

  const paxLimit = maxVal ?? section.dist.max;
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
        fill={SectionLoadColors.Bg_0_80}
      />,
      <rect
        key="80-100"
        x={x80}
        y={y}
        width={x100 - x80}
        height={h}
        fill={SectionLoadColors.Bg_80_100}
      />,
      <rect
        key="100-120"
        x={x100}
        y={y}
        width={x120 - x100}
        height={h}
        fill={SectionLoadColors.Bg_100_120}
      />,
      <rect
        key="120-200"
        x={x120}
        y={y}
        width={x200 - x120}
        height={h}
        fill={SectionLoadColors.Bg_120_200}
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
          fill={SectionLoadColors.Bg_200_plus}
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
        fill={SectionLoadColors.Bg_unknown}
      />
    );
  }

  let plot: JSX.Element | null = null;
  switch (plotType) {
    case "SimpleBox": {
      const lo = paxScale(section.dist.q5);
      const hi = paxScale(section.dist.q95);
      plot = (
        <g>
          <path
            d={`M${lo} ${margin.top} H${hi} V${
              margin.top + innerHeight
            } H${lo} Z`}
            style={{
              fill: SectionLoadColors.Fill_Range,
              fillOpacity: 0.5,
              stroke: SectionLoadColors.Stroke_Range,
              strokeOpacity: 0.4,
            }}
          />
          <path
            d={`M${paxScale(section.dist.q50)} ${margin.top} V${
              margin.top + innerHeight
            }`}
            style={{
              stroke: SectionLoadColors.Stroke_Median,
              strokeWidth: 3,
              fill: "none",
            }}
          />
        </g>
      );
      break;
    }
    case "Violin": {
      plot = (
        <ViolinPlot
          data={section.dist.pdf}
          stroke={SectionLoadColors.Stroke_BoxViolin}
          strokeWidth={2}
          fill={SectionLoadColors.Fill_BoxViolin}
          valueScale={paxScale}
          count={count}
          value={value}
          top={margin.top + 4}
          width={innerHeight - 8}
          horizontal={true}
        />
      );
      break;
    }
    case "Box": {
      plot = (
        <BoxPlot
          min={section.dist.min}
          max={section.dist.max}
          firstQuartile={section.dist.q5}
          thirdQuartile={section.dist.q95}
          median={section.dist.q50}
          stroke={SectionLoadColors.Stroke_BoxViolin}
          strokeWidth={2}
          fill={SectionLoadColors.Fill_BoxViolin}
          valueScale={paxScale}
          top={margin.top + 4}
          boxWidth={innerHeight - 8}
          horizontal={true}
        />
      );
      break;
    }
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
      {plot}
      <path
        d={`M${paxScale(section.expected_passengers)} ${
          margin.top
        } v${innerHeight}`}
        stroke={SectionLoadColors.Stroke_Expected}
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
