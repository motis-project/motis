import { AxisBottom, AxisLeft } from "@visx/axis";
import getTickFormatter from "@visx/axis/lib/utils/getTickFormatter";
import { Grid } from "@visx/grid";
import { Group } from "@visx/group";
import { LegendOrdinal } from "@visx/legend";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear, scaleOrdinal } from "@visx/scale";
import { BarStack } from "@visx/shape";
import { range } from "d3-array";
import { fromUnixTime } from "date-fns";
import { ReactElement } from "react";

import { formatTime } from "@/util/dateFormat";

export interface MetricsTypeBase {
  start_time: number;
  entries: number;
}

export type MetricsKeyBase<MetricsType> = keyof Omit<
  MetricsType,
  "start_time" | "entries"
> &
  string;

export interface MetricInfo {
  label: string;
  color: string;
}

export interface MetricsChartProps<
  MetricsType extends MetricsTypeBase,
  MetricsKey extends MetricsKeyBase<MetricsType>,
> {
  metricsData: MetricsType;
  metricsInfo: Partial<Record<MetricsKey, MetricInfo>>;
  hideUntil?: number;

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

function MetricsChart<
  MetricsType extends MetricsTypeBase,
  MetricsKey extends MetricsKeyBase<MetricsType>,
>({
  metricsData,
  metricsInfo,
  hideUntil = 0,
  width,
  height,
  margin = defaultMargin,
  axisColor = defaultAxisColor,
  backgroundColor = defaultBackgroundColor,
}: MetricsChartProps<MetricsType, MetricsKey>) {
  margin ??= defaultMargin;

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const xMax = innerWidth - yAxisWidth;
  const yMax = innerHeight - margin.top;

  const keys = Object.keys(metricsInfo) as MetricsKey[];

  if (xMax < 50 || yMax < 50 || keys.length === 0) return null;

  const indices = (metricsData[keys[0]] as number[]).map((_, i) => i);

  const hideUntilIndex =
    hideUntil >= metricsData.start_time
      ? (hideUntil - metricsData.start_time) / 60
      : -1;

  const getValue = (idx: number, key: MetricsKey) =>
    idx > hideUntilIndex ? (metricsData[key] as number[])[idx] : 0;

  const maxNumberOfMessagesPerMinute = indices.reduce(
    (max, idx) =>
      idx > hideUntilIndex
        ? Math.max(
            max,
            keys
              .map((key) => getValue(idx, key))
              .reduce((sum, val) => sum + val, 0),
          )
        : max,
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
    range: Object.values(metricsInfo).map((m) => (m as MetricInfo).color),
  });

  const formatIndexDate = (index: number) => {
    const ts = metricsData.start_time + index * 60;
    const date = new Date(fromUnixTime(ts));
    const str = formatTime(date);
    return str !== "00:00" ? str : formatTime(date, "eeeeee HH:mm");
  };

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
            value={getValue}
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
          labelFormat={(k) => metricsInfo[k]?.label ?? ""}
        />
      </div>
    </div>
  );
}

export type ResponsiveMetricsChartProps<
  MetricsType extends MetricsTypeBase,
  MetricsKey extends MetricsKeyBase<MetricsType>,
> = Omit<MetricsChartProps<MetricsType, MetricsKey>, "width" | "height">;

function ResponsiveMetricsChart<
  MetricsType extends MetricsTypeBase,
  MetricsKey extends MetricsKeyBase<MetricsType>,
>(props: ResponsiveMetricsChartProps<MetricsType, MetricsKey>): ReactElement {
  return (
    <ParentSize>
      {({ width, height }) => (
        <MetricsChart width={width} height={height} {...props} />
      )}
    </ParentSize>
  );
}

export default ResponsiveMetricsChart;
