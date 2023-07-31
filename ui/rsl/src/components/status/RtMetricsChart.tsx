import { AxisBottom, AxisLeft } from "@visx/axis";
import getTickFormatter from "@visx/axis/lib/utils/getTickFormatter";
import { Grid } from "@visx/grid";
import { Group } from "@visx/group";
import { LegendOrdinal } from "@visx/legend";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear, scaleOrdinal } from "@visx/scale";
import { BarStack } from "@visx/shape";
import { range } from "d3-array";
import { ReactElement, useMemo } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { formatTime } from "@/util/dateFormat";

export interface RtMetricsChartProps {
  metrics: RtMetrics;
  width: number;
  height: number;
  margin?: { top: number; right: number; bottom: number; left: number };
}

const dataColor1 = "#090062";
const dataColor2 = "#6c5efb";
const dataColor3 = "#f8a041";
const axisColor = "#a44afe";
const background = "#eaedff";
const defaultMargin = { top: 40, right: 20, bottom: 0, left: 20 };

interface MinuteMetrics {
  index: number;
  Sollfahrten: number;
  "Fahrt Updates": number;
  "Formation Updates": number;
}

type MetricName = "Sollfahrten" | "Fahrt Updates" | "Formation Updates";
const keys: MetricName[] = [
  "Sollfahrten",
  "Fahrt Updates",
  "Formation Updates",
];

const getIndex = (d: MinuteMetrics) => d.index;

const yAxisWidth = 50;

function RtMetricsChart({
  metrics,
  width,
  height,
  margin = defaultMargin,
}: RtMetricsChartProps) {
  margin ??= defaultMargin;

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const xMax = innerWidth - yAxisWidth;
  const yMax = innerHeight - margin.top;

  const data = useMemo(() => {
    const data: MinuteMetrics[] = [];
    for (let i = 0; i < metrics.entries; i++) {
      data.push({
        index: i,
        Sollfahrten: metrics.full_trip_schedule_messages[i],
        "Fahrt Updates": metrics.full_trip_update_messages[i],
        "Formation Updates": metrics.trip_formation_messages[i],
      });
    }
    return data;
  }, [metrics]);

  if (xMax < 50 || yMax < 50) return null;

  const timeScale = scaleBand<number>({
    domain: range(0, metrics.entries),
    range: [0, xMax],
  });
  const countScale = scaleLinear<number>({
    domain: [0, Math.max(...metrics.messages)],
    range: [yMax, 0],
    nice: true,
  });
  const colorScale = scaleOrdinal<MetricName, string>({
    domain: keys,
    range: [dataColor1, dataColor2, dataColor3],
  });

  const formatIndexDate = (index: number) =>
    formatTime(metrics.start_time + index * 60);

  // x-axis ticks for each full hour
  const firstFullHour = Math.ceil(metrics.start_time / 3600) * 3600;
  const timeTicks: number[] = [];
  for (
    let i = (firstFullHour - metrics.start_time) / 60;
    i < metrics.entries;
    i += 60
  ) {
    timeTicks.push(i);
  }

  const yTickFormat = getTickFormatter(countScale);

  return (
    <div style={{ position: "relative" }}>
      <svg width={width} height={height}>
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          fill={background}
          rx={14}
        />
        <Group top={margin.top} left={margin.left + yAxisWidth}>
          <Grid
            top={0}
            left={0}
            xScale={timeScale}
            yScale={countScale}
            width={xMax}
            height={yMax}
            stroke="black"
            strokeOpacity={0.1}
            columnTickValues={timeTicks}
          />
          <BarStack<MinuteMetrics, MetricName>
            data={data}
            keys={keys}
            x={getIndex}
            xScale={timeScale}
            yScale={countScale}
            color={colorScale}
          >
            {(barStacks) =>
              barStacks.map((barStack) =>
                barStack.bars.map((bar) => (
                  <rect
                    key={`bar-stack-${barStack.index}-${bar.index}`}
                    x={bar.x}
                    y={bar.y}
                    height={bar.height}
                    width={bar.width}
                    fill={bar.color}
                  />
                )),
              )
            }
          </BarStack>
          <AxisLeft
            scale={countScale}
            top={0}
            left={0}
            tickFormat={yTickFormat}
            stroke={axisColor}
            tickStroke={axisColor}
            tickLabelProps={{
              fill: axisColor,
              fontSize: 11,
            }}
          />
          <AxisBottom
            top={yMax}
            left={0}
            scale={timeScale}
            tickFormat={formatIndexDate}
            stroke={axisColor}
            tickStroke={axisColor}
            tickLabelProps={{
              fill: axisColor,
              fontSize: 11,
              textAnchor: "middle",
            }}
            tickValues={timeTicks}
          />
        </Group>
      </svg>
      <div
        style={{
          position: "absolute",
          top: margin.top / 2 - 10,
          width: "100%",
          display: "flex",
          justifyContent: "center",
          fontSize: "14px",
        }}
      >
        <LegendOrdinal
          scale={colorScale}
          direction="row"
          labelMargin="0 15px 0 0"
        />
      </div>
    </div>
  );
}

export type ResponsiveRtMetricsChartProps = Omit<
  RtMetricsChartProps,
  "width" | "height"
>;

function ResponsiveRtMetricsChart(
  props: ResponsiveRtMetricsChartProps,
): ReactElement {
  return (
    <ParentSize>
      {({ width, height }) => (
        <RtMetricsChart width={width} height={height} {...props} />
      )}
    </ParentSize>
  );
}

export default ResponsiveRtMetricsChart;
