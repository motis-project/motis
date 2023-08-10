import { ArrowDownTrayIcon } from "@heroicons/react/20/solid";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAtom } from "jotai";
import { CSSProperties, useRef } from "react";

import { TripId } from "@/api/protocol/motis";
import {
  PaxMonEdgeLoadInfo,
  PaxMonTripLoadInfo,
} from "@/api/protocol/motis/paxmon";

import {
  queryKeys,
  sendPaxMonGetTripLoadInfosRequest,
  usePaxMonStatusQuery,
} from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import {
  formatFileNameTime,
  formatLongDateTime,
  formatTime,
} from "@/util/dateFormat";

function getSvgLinePath(
  edges: PaxMonEdgeLoadInfo[],
  maxVal: number,
  getProp: (ef: PaxMonEdgeLoadInfo) => number,
) {
  const points = [];
  let x = 0;
  for (const ef of edges) {
    const load = getProp(ef);
    const y = 200 - Math.round((load / maxVal) * 200);
    points.push(`${x} ${y}`);
    x += 50;
    points.push(`${x} ${y}`);
  }

  if (points.length > 0) {
    return "M" + points.join(" ");
  } else {
    return "";
  }
}

function getYLabels(maxVal: number) {
  const stepSize =
    maxVal >= 20000
      ? 2000
      : maxVal >= 10000
      ? 1000
      : maxVal >= 3000
      ? 500
      : maxVal >= 1500
      ? 200
      : maxVal >= 700
      ? 100
      : 50;
  const labels = [];
  for (let pax = stepSize; pax < maxVal; pax += stepSize) {
    labels.push({
      pax,
      y: 200 - Math.round((pax / maxVal) * 200),
    });
  }
  return labels;
}

function getCurrentTimePosition(
  edges: PaxMonEdgeLoadInfo[],
  currentTime: number,
) {
  if (currentTime < edges[0].departure_current_time) {
    return -5;
  } else {
    for (const [idx, e] of edges.entries()) {
      const dep = e.departure_current_time;
      const arr = e.arrival_current_time;
      if (currentTime <= dep) {
        return idx * 50;
      } else if (currentTime <= arr) {
        const sectionPos = (currentTime - dep) / (arr - dep);
        return Math.round(idx * 50 + sectionPos * 50);
      }
    }
  }
  return edges.length * 50 + 5;
}

function getSvgBlob(svgEl: SVGSVGElement) {
  const serializer = new XMLSerializer();
  const source = serializer.serializeToString(svgEl);
  return new Blob([source], { type: "image/svg+xml;charset=utf-8" });
}

function downloadBlob(url: string, filename: string) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
}

function saveAsSVG(svgEl: SVGSVGElement | null, baseFileName: string) {
  if (!svgEl) {
    return;
  }
  const svgBlob = getSvgBlob(svgEl);
  const url = URL.createObjectURL(svgBlob);
  downloadBlob(url, baseFileName + ".svg");
}

function saveAsPNG(svgEl: SVGSVGElement | null, baseFileName: string) {
  if (!svgEl) {
    return;
  }
  const svgBlob = getSvgBlob(svgEl);
  const svgUrl = URL.createObjectURL(svgBlob);
  const svgBB = svgEl.getBoundingClientRect();
  const canvas = document.createElement("canvas");
  canvas.width = svgBB.width * 2;
  canvas.height = svgBB.height * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(svgUrl);
    const pngUrl = canvas.toDataURL("image/png");
    downloadBlob(pngUrl, baseFileName + ".png");
  };
  img.src = svgUrl;
}

function getBaseFileName(data: PaxMonTripLoadInfo, systemTime: number) {
  const parts = ["forecast", formatFileNameTime(systemTime)];
  for (const si of data.tsi.service_infos) {
    if (si.line) {
      parts.push(`${si.train_nr}-${si.category}-${si.line}`);
    } else {
      parts.push(`${si.train_nr}-${si.category}`);
    }
  }
  return parts.join("_");
}

interface TripLoadForecastChartProps {
  tripId: TripId;
  mode: "Interactive" | "Tooltip";
  onSectionClick?: (e: PaxMonEdgeLoadInfo) => void;
}

function TripLoadForecastChart({
  tripId,
  mode,
  onSectionClick,
}: TripLoadForecastChartProps): JSX.Element | null {
  const [universe] = useAtom(universeAtom);
  const { data: status } = usePaxMonStatusQuery(universe);

  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery(
    queryKeys.tripLoad(universe, tripId),
    () => sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    {
      enabled: !!status,
      placeholderData: () => {
        return universe != 0
          ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
          : undefined;
      },
    },
  );

  const svgEl = useRef<SVGSVGElement>(null);

  if (!status || !data) {
    return null;
  }

  const systemTime = status.system_time;
  const tripData = data.load_infos[0];
  const edges = tripData.edges;
  const graphWidth = edges.length * 50;

  const maxPax = edges.reduce((max, ef) => Math.max(max, ef.dist.max), 0);
  const maxCapacity = edges.reduce(
    (max, ef) => (ef.capacity ? Math.max(max, ef.capacity) : max),
    0,
  );
  const maxVal = Math.max(maxPax, maxCapacity) * 1.1;

  const background = edges.map((e, idx) => {
    const x = idx * 50;
    if (e.capacity == 0) {
      return (
        <g key={idx.toString()}>
          <rect
            x={x}
            y={0}
            width="50"
            height={200}
            stroke="#DDD"
            fill="#ffffff"
          />
        </g>
      );
    }
    const over200 = maxVal >= e.capacity * 2;
    const y200 = over200
      ? 200 - Math.round(((e.capacity * 2.0) / maxVal) * 200)
      : 0;
    const y100 = 200 - Math.round((e.capacity / maxVal) * 200);
    const y80 = 200 - Math.round(((e.capacity * 0.8) / maxVal) * 200);
    const yDarkRed = 0;
    const hDarkRed = y200;
    const yRed = y200;
    const hRed = y100 - y200;
    const yYellow = y100;
    const hYellow = y80 - y100;
    const yGreen = y80;
    const hGreen = 200 - y80;
    return (
      <g key={idx.toString()}>
        <rect
          x={x}
          y={yDarkRed}
          width="50"
          height={hDarkRed}
          stroke="#DDD"
          fill="#f57a7a"
        />
        <rect
          x={x}
          y={yRed}
          width="50"
          height={hRed}
          stroke="#DDD"
          fill="#FFCACA"
        />
        <rect
          x={x}
          y={yYellow}
          width="50"
          height={hYellow}
          stroke="#DDD"
          fill="#FFF3CA"
        />
        <rect
          x={x}
          y={yGreen}
          width="50"
          height={hGreen}
          stroke="#DDD"
          fill="#D4FFCA"
        />
      </g>
    );
  });

  const sectionDividers = edges.map((_, idx) => (
    <path key={idx.toString()} stroke="#DDD" d={`M${(idx + 1) * 50} 0v200`} />
  ));

  const overCapProbs = edges.map((e, idx) => {
    const style: CSSProperties = { fontSize: "8px", fill: "#000" };
    let text = "";
    if (e.capacity != 0) {
      text = `${(e.prob_over_capacity * 100).toFixed(0)}%`;
      if (text === "0%") {
        style.fill = "#999";
      }
    }
    return (
      <text
        key={idx.toString()}
        x={idx * 50 + 25}
        y="8"
        textAnchor="middle"
        style={style}
      >
        {text}
      </text>
    );
  });

  const names = [
    ...new Set(
      tripData.tsi.service_infos.map(
        (si) =>
          `${si.category} ${si.train_nr}` +
          (si.line ? ` [Linie ${si.line}]` : ""),
      ),
    ),
  ];
  const title = `${names.join(", ")}, Vorhersage vom ${formatLongDateTime(
    systemTime,
  )}`;

  const baseFileName = getBaseFileName(tripData, systemTime);

  const spreadTopPoints = [];
  const spreadBottomPoints = [];
  let x = 0;
  for (const ef of edges) {
    const topLoad = ef.dist.q95;
    const bottomLoad = ef.dist.q5;
    const topY = 200 - Math.round((topLoad / maxVal) * 200);
    const bottomY = 200 - Math.round((bottomLoad / maxVal) * 200);
    spreadTopPoints.push(`${x} ${topY}`);
    spreadBottomPoints.unshift(`${x} ${bottomY}`);
    x += 50;
    spreadTopPoints.push(`${x} ${topY}`);
    spreadBottomPoints.unshift(`${x} ${bottomY}`);
  }
  const spreadPath = (
    <path
      d={`M${spreadTopPoints.join(" ")} ${spreadBottomPoints.join(" ")}z`}
      style={{ fill: "#B2B5FE", fillOpacity: 0.4, stroke: "#797EFF" }}
    />
  );

  const expectedPath = (
    <path
      d={getSvgLinePath(edges, maxVal, (ef) => ef.expected_passengers)}
      style={{ stroke: "#333", strokeDasharray: 2, fill: "none" }}
    />
  );

  const medianPath = (
    <path
      d={getSvgLinePath(edges, maxVal, (ef) => ef.dist.q50)}
      style={{ stroke: "#3038FF", strokeWidth: 2, fill: "none" }}
    />
  );

  const yLabels = getYLabels(maxVal).map((label) => (
    <text
      x="-2"
      y={label.y + 4}
      textAnchor="end"
      style={{ fontSize: "8px", fill: "#333" }}
      key={label.pax}
    >
      {label.pax}
    </text>
  ));

  const stationNameLabels = [];
  const scheduleTimeLabels = [];
  const currentTimeLabels = [];
  const stops = [edges[0].from].concat(edges.map((ef) => ef.to));
  for (const [idx, station] of stops.entries()) {
    const x = idx * 50;
    const arrivalScheduleTime =
      idx > 0 ? edges[idx - 1].arrival_schedule_time : null;
    const arrivalCurrentTime =
      idx > 0 ? edges[idx - 1].arrival_current_time : null;
    const arrivalDelayed =
      idx > 0 &&
      edges[idx - 1].arrival_current_time >
        edges[idx - 1].arrival_schedule_time;
    const departureScheduleTime =
      idx < edges.length ? edges[idx].departure_schedule_time : null;
    const departureCurrentTime =
      idx < edges.length ? edges[idx].departure_current_time : null;
    const departureDelayed =
      idx < edges.length &&
      edges[idx].departure_current_time > edges[idx].departure_schedule_time;

    stationNameLabels.push(
      <text
        x="0"
        y="0"
        textAnchor="end"
        style={{ fontSize: "8px", fill: "#000" }}
        transform={`translate(${x} 210) rotate(-60 0 0)`}
        key={idx}
      >
        {station.name}
        <title>{station.id}</title>
      </text>,
    );

    if (arrivalScheduleTime && arrivalCurrentTime) {
      if (arrivalScheduleTime !== arrivalCurrentTime) {
        scheduleTimeLabels.push(
          <text
            x={x - 2}
            y="190"
            textAnchor="end"
            style={{ fontSize: "6px", fill: "#777" }}
            key={`${idx}.sched.arr`}
          >
            {formatTime(arrivalScheduleTime)}
          </text>,
        );
      }
      currentTimeLabels.push(
        <text
          x={x - 2}
          y="198"
          textAnchor="end"
          style={{
            fontSize: "6px",
            fill: arrivalDelayed ? "#d60000" : "#008600",
          }}
          key={`${idx}.curr.arr`}
        >
          {formatTime(arrivalCurrentTime)}
        </text>,
      );
    }
    if (departureScheduleTime && departureCurrentTime) {
      if (departureScheduleTime !== departureCurrentTime) {
        scheduleTimeLabels.push(
          <text
            x={x + 2}
            y="190"
            textAnchor="start"
            style={{ fontSize: "6px", fill: "#777" }}
            key={`${idx}.sched.dep`}
          >
            {formatTime(departureScheduleTime)}
          </text>,
        );
      }
      currentTimeLabels.push(
        <text
          x={x + 2}
          y="198"
          textAnchor="start"
          style={{
            fontSize: "6px",
            fill: departureDelayed ? "#d60000" : "#008600",
          }}
          key={`${idx}.curr.dep`}
        >
          {formatTime(departureCurrentTime)}
        </text>,
      );
    }
  }

  const outerBorder = (
    <rect
      x="0"
      y="0"
      width={graphWidth}
      height="200"
      stroke="#333"
      fill="transparent"
    />
  );

  const currentTimePosition = getCurrentTimePosition(edges, systemTime);
  const currentTimeIndicator = (
    <path
      d={`M${currentTimePosition} 201 l2 4 l-4 0 z`}
      style={{ fill: "#777" }}
    />
  );

  const clickRegions = onSectionClick
    ? edges.map((e, idx) => {
        return (
          <rect
            key={idx.toString()}
            x={idx * 50}
            y="0"
            width="50"
            height="200"
            fill="transparent"
            className="cursor-pointer"
            onClick={() => {
              onSectionClick(e);
            }}
          />
        );
      })
    : [];

  const chart = (
    <svg
      ref={svgEl}
      viewBox={`-100 -15 ${120 + graphWidth} 335`}
      className="max-h-[42rem] mx-auto mt-2"
    >
      <g style={{ fontFamily: "Arial, Helvetica, sans-serif" }}>
        <g>{background}</g>
        <g>{sectionDividers}</g>
        <g>
          <path stroke="#DDD" d={`M0 10h${graphWidth}`} />
          {overCapProbs}
        </g>
        {spreadPath}
        {expectedPath}
        {medianPath}
        {outerBorder}
        <text
          x={graphWidth / 2}
          y="-5"
          textAnchor="middle"
          style={{ fontSize: "8px", fill: "#333" }}
        >
          {title}
        </text>
        <g>{yLabels}</g>
        <g>
          {stationNameLabels}
          {scheduleTimeLabels}
          {currentTimeLabels}
        </g>
        {currentTimeIndicator}
        <g>{clickRegions}</g>
      </g>
    </svg>
  );

  if (mode == "Interactive") {
    return (
      <div>
        <div>{chart}</div>
        <div className="flex justify-center gap-2 my-2">
          <button
            type="button"
            onClick={() => saveAsSVG(svgEl.current, baseFileName)}
            className="flex items-center bg-db-red-500 px-3 py-1 rounded text-white text-sm hover:bg-db-red-600"
          >
            <ArrowDownTrayIcon className="w-5 h-5 mr-2" aria-hidden="true" />
            SVG
          </button>
          <button
            type="button"
            onClick={() => saveAsPNG(svgEl.current, baseFileName)}
            className="flex items-center bg-db-red-500 px-3 py-1 rounded text-white text-sm hover:bg-db-red-600"
          >
            <ArrowDownTrayIcon className="w-5 h-5 mr-2" aria-hidden="true" />
            PNG
          </button>
        </div>
      </div>
    );
  } else {
    return chart;
  }
}

export default TripLoadForecastChart;
