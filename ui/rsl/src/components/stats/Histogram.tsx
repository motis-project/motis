import { AxisBottom, AxisLeft } from "@visx/axis";
import { TickFormatter } from "@visx/axis/lib/types";
import getTickFormatter from "@visx/axis/lib/utils/getTickFormatter";
import { Grid } from "@visx/grid";
import { Group } from "@visx/group";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear, scaleSymlog } from "@visx/scale";
import { Bar } from "@visx/shape";
import { range } from "d3-array";

import { PaxMonHistogram } from "@/api/protocol/motis/paxmon";

const defaultMargin = {
  top: 4,
  right: 4,
  bottom: 4,
  left: 4,
};

type ScaleType = "linear" | "symlog";

type HistogramProps = {
  data: PaxMonHistogram;
  width: number;
  height: number;
  margin?: typeof defaultMargin;
  bgClass?: string;
  barClass?: string;
  yScaleType?: ScaleType;
  xTickFormat?: TickFormatter<number>;
};

const xAxisHeight = 25;
const yAxisWidth = 50;

function Histogram({
  data,
  width,
  height,
  margin = defaultMargin,
  bgClass = "fill-gray-200",
  barClass = "fill-blue-800",
  yScaleType = "linear",
  xTickFormat,
}: HistogramProps): JSX.Element {
  margin ??= defaultMargin;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const xMax = innerWidth - yAxisWidth;
  const yMax = innerHeight - xAxisHeight;

  if (xMax < 50 || yMax < 50) {
    return <></>;
  }

  const xScale = scaleBand<number>({
    domain: range(data.min_value, data.max_value),
    range: [0, xMax],

    paddingInner: 0.1,
  });

  const yScale =
    yScaleType === "symlog"
      ? scaleSymlog({
          domain: [0, data.max_count],
          range: [yMax, 0],
        })
      : scaleLinear({
          domain: [0, data.max_count],
          range: [yMax, 0],
        });

  xTickFormat ??= getTickFormatter(xScale);

  const barWidth = xScale.bandwidth();

  return (
    <svg width={width} height={height}>
      <rect width={width} height={height} className={bgClass} rx={4} />
      <Grid
        top={margin.top}
        left={margin.left + yAxisWidth}
        width={xMax}
        height={yMax}
        xScale={xScale}
        yScale={yScale}
        stroke="black"
        strokeOpacity={0.1}
      />
      <Group top={margin.top} left={margin.left + yAxisWidth}>
        {data.counts.map((count, idx) => {
          const value = data.min_value + idx;
          const barHeight = yMax - yScale(count) ?? 0;
          const barX = xScale(value);
          const barY = yMax - barHeight;
          return (
            <Bar
              key={`bar-${value}`}
              x={barX}
              y={barY}
              width={barWidth}
              height={barHeight}
              className={barClass}
            />
          );
        })}
      </Group>
      <AxisLeft
        scale={yScale}
        top={margin.top}
        left={margin.left + yAxisWidth}
      />
      <AxisBottom
        scale={xScale}
        top={margin.top + yMax}
        left={margin.left + yAxisWidth}
        tickFormat={xTickFormat}
      />
    </svg>
  );
}

function ResponsiveHistogram(
  props: Omit<HistogramProps, "width" | "height">
): JSX.Element {
  return (
    <ParentSize>
      {({ width, height }) => (
        <Histogram width={width} height={height} {...props} />
      )}
    </ParentSize>
  );
}

export default ResponsiveHistogram;
