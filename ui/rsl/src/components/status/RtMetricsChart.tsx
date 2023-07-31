import { AxisBottom, AxisLeft } from "@visx/axis";
import getTickFormatter from "@visx/axis/lib/utils/getTickFormatter";
import { Grid } from "@visx/grid";
import { Group } from "@visx/group";
import { LegendOrdinal } from "@visx/legend";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear, scaleOrdinal } from "@visx/scale";
import { BarStack } from "@visx/shape";
import { range } from "d3-array";
import { ReactElement } from "react";

import { RtMetrics } from "@/api/protocol/motis/rt";

import { formatTime } from "@/util/dateFormat";

export type MetricsKeyBase = keyof Omit<RtMetrics, "start_time" | "entries">;

export interface MetricInfo {
  label: string;
  color: string;
}

export interface RtMetricsChartProps<MetricsKey extends MetricsKeyBase> {
  metricsData: RtMetrics;
  metricsInfo: Record<MetricsKey, MetricInfo>;

  width: number;
  height: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  axisColor?: string;
  backgroundColor?: string;
}

const defaultAxisColor = "#000000";
const defaultBackgroundColor = "#F0F3F5";
const defaultMargin = { top: 40, right: 20, bottom: 0, left: 20 };

const getIndex = (i: number) => i;

const yAxisWidth = 50;

function RtMetricsChart<MetricsKey extends MetricsKeyBase>({
  metricsData,
  metricsInfo,
  width,
  height,
  margin = defaultMargin,
  axisColor = defaultAxisColor,
  backgroundColor = defaultBackgroundColor,
}: RtMetricsChartProps<MetricsKey>) {
  margin ??= defaultMargin;

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const xMax = innerWidth - yAxisWidth;
  const yMax = innerHeight - margin.top;

  if (xMax < 50 || yMax < 50) return null;

  const keys = Object.keys(metricsInfo) as MetricsKey[];
  const indices = metricsData.messages.map((_, i) => i);

  const maxNumberOfMessagesPerMinute = indices.reduce(
    (max, idx) =>
      Math.max(
        max,
        keys
          .map((key) => metricsData[key][idx])
          .reduce((sum, val) => sum + val, 0),
      ),
    0,
  );

  const timeScale = scaleBand<number>({
    domain: range(0, metricsData.entries),
    range: [0, xMax],
  });
  const countScale = scaleLinear<number>({
    domain: [0, maxNumberOfMessagesPerMinute],
    range: [yMax, 0],
    nice: true,
  });
  const colorScale = scaleOrdinal<MetricsKey, string>({
    domain: keys,
    range: Object.values<MetricInfo>(metricsInfo).map((m) => m.color),
  });

  const formatIndexDate = (index: number) =>
    formatTime(metricsData.start_time + index * 60);

  // x-axis ticks for each full hour
  const firstFullHour = Math.ceil(metricsData.start_time / 3600) * 3600;
  const timeTicks: number[] = [];
  for (
    let i = (firstFullHour - metricsData.start_time) / 60;
    i < metricsData.entries;
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
          fill={backgroundColor}
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
          <BarStack<number, MetricsKey>
            data={indices}
            keys={keys}
            value={(idx, key) => metricsData[key][idx]}
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
          labelFormat={(k) => metricsInfo[k].label}
        />
      </div>
    </div>
  );
}

export type ResponsiveRtMetricsChartProps<MetricsKey extends MetricsKeyBase> =
  Omit<RtMetricsChartProps<MetricsKey>, "width" | "height">;

function ResponsiveRtMetricsChart<MetricsKey extends MetricsKeyBase>(
  props: ResponsiveRtMetricsChartProps<MetricsKey>,
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
