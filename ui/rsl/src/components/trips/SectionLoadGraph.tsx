import { AxisBottom } from "@visx/axis";
import { GridColumns } from "@visx/grid";
import { ParentSize } from "@visx/responsive";
import { scaleLinear } from "@visx/scale";
import { BoxPlot, ViolinPlot } from "@visx/stats";
import { CSSProperties, HTMLAttributes, ReactNode } from "react";

import {
  PaxMonEdgeLoadInfo,
  PaxMonPdfEntry,
} from "@/api/protocol/motis/paxmon";

import { SectionLoadColors } from "@/util/colors";

import { cn } from "@/lib/utils";

export type SectionLoadGraphPlotType = "SimpleBox" | "Violin" | "Box";

const defaultMargin = {
  top: 0,
  right: 10,
  bottom: 25,
  left: 10,
};

interface SectionLoadGraphProps {
  section: PaxMonEdgeLoadInfo;
  width: number;
  height: number;
  maxVal?: number | undefined;
  margin?: typeof defaultMargin;
  plotType?: SectionLoadGraphPlotType;
}

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
      />,
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
        />,
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
      />,
    );
  }

  const simpleGraph = plotType == "SimpleBox";
  const lo = paxScale(simpleGraph ? section.dist.q5 : section.dist.min);
  const hi = paxScale(simpleGraph ? section.dist.q95 : section.dist.max);

  let plot: JSX.Element | null = null;
  switch (plotType) {
    case "SimpleBox": {
      plot = (
        <g>
          <path
            d={`M${lo} ${margin.top + 2} H${hi} V${
              margin.top + innerHeight - 2
            } H${lo} Z`}
            style={{
              fill: SectionLoadColors.Fill_Range,
              stroke: SectionLoadColors.Stroke_Range,
              strokeWidth: 2,
            }}
          />
          <path
            d={`M${paxScale(section.dist.q50)} ${margin.top + 2} V${
              margin.top + innerHeight - 2
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

  const tooltipStyle: CSSProperties = { top: margin.top + 2 };
  if (hi > width - 200) {
    tooltipStyle.right = width - lo + 10;
  } else {
    tooltipStyle.left = hi + 10;
  }

  return (
    <div className="group relative">
      <svg width="100%" height={height}>
        <g>{bgSections}</g>
        <rect
          x={margin.left}
          y={margin.top}
          width={innerWidth}
          height={innerHeight}
          fill="#fff"
          className="opacity-0 group-hover:opacity-20"
        />
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
          stroke={SectionLoadColors.Stroke_Expected1}
          strokeDasharray={2}
          strokeWidth={2}
        />
        <path
          d={`M${paxScale(section.expected_passengers)} ${margin.top + 2} v${
            innerHeight - 2
          }`}
          stroke={SectionLoadColors.Stroke_Expected2}
          strokeDasharray={2}
          strokeWidth={2}
        />
        <AxisBottom scale={paxScale} top={margin.top + innerHeight} />
      </svg>
      <div
        className="absolute hidden group-hover:block z-10 pointer-events-none w-40
         bg-white text-black shadow-lg rounded p-1 text-xs opacity-95"
        style={tooltipStyle}
      >
        <table className="w-full">
          <tbody>
            {!simpleGraph && (
              <TooltipRow pax={section.dist.min} section={section}>
                Minimum
              </TooltipRow>
            )}
            <TooltipRow pax={section.dist.q5} section={section}>
              5% Quantil
            </TooltipRow>
            <TooltipRow pax={section.dist.q50} section={section}>
              Median
            </TooltipRow>
            <TooltipRow pax={section.dist.q95} section={section}>
              95% Quantil
            </TooltipRow>
            {!simpleGraph && (
              <TooltipRow pax={section.dist.max} section={section}>
                Maximum
              </TooltipRow>
            )}
            <TooltipRow
              pax={section.expected_passengers}
              section={section}
              className="border-y-2 border-gray-300"
            >
              Planmäßig
            </TooltipRow>
            {!simpleGraph && (
              <>
                <tr>
                  <td>Spannbreite (Min–Max)</td>
                  <td className="text-right">
                    {section.dist.max - section.dist.min}
                  </td>
                </tr>
                <tr>
                  <td>Spannbreite (5%–95%)</td>
                  <td className="text-right">
                    {section.dist.q95 - section.dist.q5}
                  </td>
                </tr>
              </>
            )}
            <tr>
              <td>Kapazität</td>
              <td className="text-right">
                {section.capacity_type === "Known"
                  ? section.capacity
                  : "Unbekannt"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

type TooltipRowProps = {
  pax: number;
  section: PaxMonEdgeLoadInfo;
  children: ReactNode;
} & HTMLAttributes<HTMLTableRowElement>;

function TooltipRow({ pax, section, children, ...rest }: TooltipRowProps) {
  return (
    <tr {...rest}>
      <td>{children}</td>
      <td className={cn("text-right", getTooltipTextClass(pax, section))}>
        {pax}
      </td>
    </tr>
  );
}

function getTooltipTextClass(pax: number, section: PaxMonEdgeLoadInfo): string {
  if (section.capacity_type == "Known") {
    const load = pax / section.capacity;
    if (load <= 0.8) {
      return "text-green-800";
    } else if (load <= 1.0) {
      return "text-yellow-500";
    } else if (load <= 1.2) {
      return "text-orange-600";
    } else if (load <= 2.0) {
      return "text-red-600";
    } else {
      return "text-red-900";
    }
  } else {
    return "text-black";
  }
}

function ResponsiveSectionLoadGraph(
  props: Omit<SectionLoadGraphProps, "width" | "height">,
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
