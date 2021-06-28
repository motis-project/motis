import { useRef } from "react";

import {
  formatTime,
  formatDateTime,
  formatFileNameTime,
} from "./util/dateFormat";

function getSvgLinePath(edges, maxVal, prop) {
  let points = [];
  let x = 0;
  for (const ef of edges) {
    const load = ef[prop] || 0;
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

function getYLabels(maxVal) {
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
  let labels = [];
  for (let pax = stepSize; pax < maxVal; pax += stepSize) {
    labels.push({
      pax,
      y: 200 - Math.round((pax / maxVal) * 200),
    });
  }
  return labels;
}

function getCurrentTimePosition(edges, currentTime) {
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

function getSvgBlob(svgEl) {
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgEl);
  const css = document.getElementById("svgStyle").outerHTML;
  source = source.replace("<g", css + "<g");
  return new Blob([source], { type: "image/svg+xml;charset=utf-8" });
}

function downloadBlob(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
}

function saveAsSVG(svgEl, baseFileName) {
  const svgBlob = getSvgBlob(svgEl);
  const url = URL.createObjectURL(svgBlob);
  downloadBlob(url, baseFileName + ".svg");
}

function saveAsPNG(svgEl, baseFileName) {
  const svgBlob = getSvgBlob(svgEl);
  const svgUrl = URL.createObjectURL(svgBlob);
  const svgBB = svgEl.getBoundingClientRect();
  const canvas = document.createElement("canvas");
  canvas.width = svgBB.width * 2;
  canvas.height = svgBB.height * 2;
  const ctx = canvas.getContext("2d");
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

function getBaseFileName(data, systemTime) {
  let parts = ["forecast", formatFileNameTime(systemTime)];
  for (const si of data.tsi.service_infos) {
    if (si.line) {
      parts.push(`${si.train_nr}-${si.category}-${si.line}`);
    } else {
      parts.push(`${si.train_nr}-${si.category}`);
    }
  }
  return parts.join("_");
}

function TripLoadForecastChart(props) {
  const svgEl = useRef(null);

  const data = props.data;
  const edges = data?.edges;
  const systemTime = props.systemTime;
  if (!edges || !systemTime) {
    return null;
  }

  const allEdgesHaveCapacity = edges.every((e) => e.capacity);
  if (!allEdgesHaveCapacity) {
    return <div className="text-red-500">Missing capacity information</div>;
  }

  const graphWidth = edges.length * 50;

  const maxPax = edges.reduce((max, ef) => Math.max(max, ef.max_pax), 0);
  const maxCapacity = edges.reduce(
    (max, ef) => (ef.capacity ? Math.max(max, ef.capacity) : max),
    0
  );
  const maxVal = Math.max(maxPax, maxCapacity) * 1.1;

  const background = edges.map((e, idx) => {
    const over200 = maxVal >= e.capacity * 2;
    const y200 = over200
      ? 200 - Math.round(((e.capacity * 2.0) / maxVal) * 200)
      : 0;
    const y100 = 200 - Math.round((e.capacity / maxVal) * 200);
    const y80 = 200 - Math.round(((e.capacity * 0.8) / maxVal) * 200);
    const x = idx * 50;
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
    let classes = ["over-cap-prob"];
    let text = "";
    if (e.p_load_gt_100 !== undefined) {
      text = `${(e.p_load_gt_100 * 100).toFixed(0)}%`;
      if (text === "0%") {
        classes.push("zero");
      }
    }
    return (
      <text
        key={idx.toString()}
        x={idx * 50 + 25}
        y="8"
        textAnchor="middle"
        className={classes.join(" ")}
      >
        {text}
      </text>
    );
  });

  const names = [
    ...new Set(
      data.tsi.service_infos.map((si) =>
        si.line ? `${si.name} [${si.train_nr}]` : si.name
      )
    ),
  ];
  const title = `${names.join(", ")} (${data.tsi.primary_station.name} - ${
    data.tsi.secondary_station.name
  }), Vorhersage vom ${formatDateTime(systemTime)}`;

  const baseFileName = getBaseFileName(data, systemTime);

  let spreadTopPoints = [];
  let spreadBottomPoints = [];
  let x = 0;
  for (const ef of edges) {
    const topLoad = ef.q_95;
    const bottomLoad = ef.q_5;
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
      className="spread"
    />
  );

  const expectedPath = (
    <path
      d={getSvgLinePath(edges, maxVal, "expected_passengers")}
      className="planned"
    />
  );

  const medianPath = (
    <path d={getSvgLinePath(edges, maxVal, "q_50")} className="median" />
  );

  const yLabels = getYLabels(maxVal).map((label) => (
    <text
      x="-2"
      y={label.y + 4}
      textAnchor="end"
      className="legend"
      key={label.pax}
    >
      {label.pax}
    </text>
  ));

  let stationNameLabels = [];
  let scheduleTimeLabels = [];
  let currentTimeLabels = [];
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
        className="legend station"
        transform={`translate(${x} 210) rotate(-60 0 0)`}
        key={idx}
      >
        {station.name}
        <title>{station.eva}</title>
      </text>
    );

    if (arrivalScheduleTime) {
      if (arrivalScheduleTime !== arrivalCurrentTime) {
        scheduleTimeLabels.push(
          <text
            x={x - 2}
            y="190"
            textAnchor="end"
            className="time schedule"
            key={`${idx}.sched.arr`}
          >
            {formatTime(arrivalScheduleTime)}
          </text>
        );
      }
      currentTimeLabels.push(
        <text
          x={x - 2}
          y="198"
          textAnchor="end"
          className={`time current${arrivalDelayed ? " delayed" : ""}`}
          key={`${idx}.curr.arr`}
        >
          {formatTime(arrivalCurrentTime)}
        </text>
      );
    }
    if (departureScheduleTime) {
      if (departureScheduleTime !== departureCurrentTime) {
        scheduleTimeLabels.push(
          <text
            x={x + 2}
            y="190"
            textAnchor="start"
            className="time schedule"
            key={`${idx}.sched.dep`}
          >
            {formatTime(departureScheduleTime)}
          </text>
        );
      }
      currentTimeLabels.push(
        <text
          x={x + 2}
          y="198"
          textAnchor="start"
          className={`time current${departureDelayed ? " delayed" : ""}`}
          key={`${idx}.curr.dep`}
        >
          {formatTime(departureCurrentTime)}
        </text>
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
      className="current-time-indicator"
    />
  );

  return (
    <div>
      <svg
        ref={svgEl}
        viewBox={`-100 -15 ${120 + graphWidth} 335`}
        style={{ height: "90vh", width: "100%", marginTop: "10px" }}
      >
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
        <text x={graphWidth / 2} y="-5" textAnchor="middle" className="legend">
          {title}
        </text>
        <g>{yLabels}</g>
        <g>
          {stationNameLabels}
          {scheduleTimeLabels}
          {currentTimeLabels}
        </g>
        {currentTimeIndicator}
      </svg>
      <div className="flex flex-row justify-center items-center space-x-2 m-2">
        <button
          className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
          onClick={() => saveAsSVG(svgEl.current, baseFileName)}
        >
          Save as SVG
        </button>
        <button
          className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
          onClick={() => saveAsPNG(svgEl.current, baseFileName)}
        >
          Save as PNG
        </button>
      </div>
    </div>
  );
}

export default TripLoadForecastChart;
