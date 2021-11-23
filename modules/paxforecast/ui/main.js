const worker = new Worker("worker.js");

const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "long",
});

const timeFormat = new Intl.DateTimeFormat(undefined, {
  timeStyle: "short",
});

let tripData = [];

const App = {
  data() {
    return {
      selectedScenario: "",
      selectedTrip: "",
      loadingText: "",
      scenarios: [],
      trips: [],
      selectedTripData: null,
    };
  },
  computed: {
    tripSectionCount() {
      return this.selectedTripData ? this.selectedTripData.edges.length : 0;
    },
    selectedTripName() {
      let name = "";
      const td = this.selectedTripData;
      if (td) {
        name = td.tripDisplayName || "";
        if (td.primaryStation?.name && td.secondaryStation?.name) {
          name += ` (${td.primaryStation.name} - ${td.secondaryStation.name})`;
        }
      }
      return name;
    },
    selectedScenarioTime() {
      const td = this.selectedTripData;
      return td
        ? dateTimeFormat.format(new Date(td.line.systemTime * 1000))
        : "N/A";
    },
    selectedSvgBaseFileName() {
      const td = this.selectedTripData;
      if (!td) {
        return "forecast";
      }
      let parts = ["forecast"];
      const systemTime = new Date(td.line.systemTime * 1000);
      parts.push(
        `${systemTime.getFullYear()}-${(
          "00" +
          (systemTime.getMonth() + 1)
        ).slice(-2)}-${("00" + systemTime.getDate()).slice(-2)}`
      );
      parts.push(
        `${("00" + systemTime.getHours()).slice(-2)}${(
          "00" + systemTime.getMinutes()
        ).slice(-2)}`
      );
      if (td.serviceInfos && td.serviceInfos.length > 0) {
        for (const si of td.serviceInfos) {
          if (si.line) {
            parts.push(`${si.train_nr}-${si.category}-${si.line}`);
          } else {
            parts.push(`${si.train_nr}-${si.category}`);
          }
        }
      } else {
        parts.push(td.trip.train_nr);
      }
      return parts.join("_");
    },
    svgGraphWidth() {
      return this.tripSectionCount * 50;
    },
    svgMidpointX() {
      return this.svgGraphWidth / 2;
    },
    svgViewBox() {
      return `-100 -15 ${120 + this.svgGraphWidth} 335`;
    },
    svgDividers() {
      if (!this.selectedTripData) {
        return [];
      }
      return this.selectedTripData.edges.map(
        (_, idx) => `M${(idx + 1) * 50} 0v200`
      );
    },
    svgMaxPaxOrCap() {
      return this.selectedTripData
        ? Math.max(
            this.selectedTripData.maxPax,
            this.selectedTripData.maxCapacity
          ) * 1.1
        : 100;
    },
    svgEdgeInfos() {
      if (!this.selectedTripData) {
        return [];
      }
      const maxVal = this.svgMaxPaxOrCap;
      return this.selectedTripData.edges.map((e, idx) => {
        const over200 = maxVal >= e.capacity * 2;
        const y200 = over200
          ? 200 - Math.round(((e.capacity * 2.0) / maxVal) * 200)
          : 0;
        const y100 = 200 - Math.round((e.capacity / maxVal) * 200);
        const y80 = 200 - Math.round(((e.capacity * 0.8) / maxVal) * 200);
        return {
          capacity: e.capacity,
          x: idx * 50,
          yDarkRed: 0,
          hDarkRed: y200,
          yRed: y200,
          hRed: y100 - y200,
          yYellow: y100,
          hYellow: y80 - y100,
          yGreen: y80,
          hGreen: 200 - y80,
        };
      });
    },
    svgOverCapProbs() {
      if (!this.selectedTripData) {
        return [];
      }
      return this.selectedTripData.edges.map((e, idx) => {
        let classes = ["over-cap-prob"];
        let text = "";
        if (e.p_load_gt_100 !== undefined) {
          text = `${(e.p_load_gt_100 * 100).toFixed(0)}%`;
          if (text === "0%") {
            classes.push("zero");
          }
        }
        return {
          x: idx * 50 + 25,
          classes,
          text,
        };
      });
    },
    svgMedianPath() {
      return getSvgLinePath(this.selectedTripData, "q_50");
    },
    svgExpectedPath() {
      return getSvgLinePath(this.selectedTripData, "expected_passengers");
    },
    svgSpreadPath() {
      let topPoints = [];
      let bottomPoints = [];
      if (this.selectedTripData && this.selectedTripData.allEdgesHaveCapacity) {
        const maxVal = this.svgMaxPaxOrCap;
        let x = 0;
        for (const ef of this.selectedTripData.edges) {
          const topLoad = ef.q_95;
          const bottomLoad = ef.q_5;
          const topY = 200 - Math.round((topLoad / maxVal) * 200);
          const bottomY = 200 - Math.round((bottomLoad / maxVal) * 200);
          topPoints.push(`${x} ${topY}`);
          bottomPoints.unshift(`${x} ${bottomY}`);
          x += 50;
          topPoints.push(`${x} ${topY}`);
          bottomPoints.unshift(`${x} ${bottomY}`);
        }
      }

      if (topPoints.length > 0 && bottomPoints.length > 0) {
        return "M" + topPoints.join(" ") + " " + bottomPoints.join(" ") + "z";
      } else {
        return "";
      }
    },
    svgYLabels() {
      if (!this.selectedTripData) {
        return [];
      }
      const maxVal = this.svgMaxPaxOrCap;
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
    },
    svgStations() {
      if (!this.selectedTripData) {
        return [];
      }
      const edges = this.selectedTripData.edges;
      return [edges[0].from]
        .concat(edges.map((ef) => ef.to))
        .map((station, idx) => {
          return {
            x: idx * 50,
            eva: station.id,
            name: station.name,
            arrivalScheduleTime:
              idx > 0 ? edges[idx - 1].arrival_schedule_time : null,
            arrivalCurrentTime:
              idx > 0 ? edges[idx - 1].arrival_current_time : null,
            arrivalDelayed:
              idx > 0 &&
              edges[idx - 1].arrival_current_time >
                edges[idx - 1].arrival_schedule_time,
            departureScheduleTime:
              idx < edges.length ? edges[idx].departure_schedule_time : null,
            departureCurrentTime:
              idx < edges.length ? edges[idx].departure_current_time : null,
            departureDelayed:
              idx < edges.length &&
              edges[idx].departure_current_time >
                edges[idx].departure_schedule_time,
          };
        });
    },
    svgCurrentTimePosition() {
      const td = this.selectedTripData;
      if (!td) {
        return 0;
      }
      const edges = td.edges;
      const currentTime = td.line.systemTime;
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
    },
    svgCurrentTimeIndicatorPath() {
      const x = this.svgCurrentTimePosition;
      return `M${x} 201 l2 4 l-4 0 z`;
    },
  },
  methods: {
    formatDateTime(timestamp) {
      return dateTimeFormat.format(new Date(timestamp * 1000));
    },
    formatTime(timestamp) {
      return timestamp ? timeFormat.format(new Date(timestamp * 1000)) : "";
    },
    saveSvg() {
      const svgBlob = getSvgBlob();
      const url = URL.createObjectURL(svgBlob);
      downloadBlob(url, this.selectedSvgBaseFileName + ".svg");
    },
    savePng() {
      const svg = document.getElementById("forecastSvg");
      const svgBlob = getSvgBlob();
      const svgUrl = URL.createObjectURL(svgBlob);
      const svgBB = svg.getBoundingClientRect();
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
        downloadBlob(pngUrl, this.selectedSvgBaseFileName + ".png");
      };
      img.src = svgUrl;
    },
    showNextTrip(offset) {
      if (this.trips.length === 0) {
        return;
      }
      if (this.selectedTrip === "") {
        this.selectedTrip = 0;
        return;
      }
      this.selectedTrip += offset;
    },
    showNextScenario(offset) {
      if (this.scenarios.length === 0) {
        return;
      }
      if (this.selectedScenario === "") {
        this.selectedScenario = 0;
        return;
      }
      this.selectedScenario += offset;
    },
    findByTrainNr() {
      const input = prompt("Train nr:", "");
      if (!input) {
        return;
      }
      const train_nr = parseInt(input);
      if (!train_nr) {
        return;
      }
      this.selectedScenario = "";
      tripData = [];
      this.trips = [];
      this.selectedTrip = "";
      vm.loadingText = "Finding trips...";
      worker.postMessage({
        op: "findInterestingTrips",
        attr: "trainNr",
        train_nr,
      });
    },
  },
  watch: {
    selectedScenario(newScenario) {
      if (this.scenarios.length === 0 || newScenario === "") {
        return;
      }
      const scenario = this.scenarios[parseInt(newScenario)];
      tripData = [];
      this.trips = [];
      this.selectedTrip = "";
      vm.loadingText = "Loading forecast info...";
      worker.postMessage(Vue.toRaw(scenario.msg));
    },
    selectedTrip(newTrip) {
      if (newTrip === "") {
        this.selectedTripData = null;
        return;
      }
      this.selectedTripData = tripData[parseInt(newTrip)];
      console.log(tripData[parseInt(newTrip)]);
    },
  },
};

function getSvgLinePath(td, prop) {
  let points = [];
  if (td && td.allEdgesHaveCapacity) {
    const maxVal = vm.svgMaxPaxOrCap;
    let x = 0;
    for (const ef of td.edges) {
      const load = ef[prop] || 0;
      const y = 200 - Math.round((load / maxVal) * 200);
      points.push(`${x} ${y}`);
      x += 50;
      points.push(`${x} ${y}`);
    }
  }

  if (points.length > 0) {
    return "M" + points.join(" ");
  } else {
    return "";
  }
}

function getSvgBlob() {
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(
    document.getElementById("forecastSvg")
  );
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

const app = Vue.createApp(App);
const vm = app.mount("#app");

worker.addEventListener("message", (e) => {
  switch (e.data.op) {
    case "fileLoadProgress": {
      vm.loadingText = `Loading file... ${(
        (e.data.offset / e.data.size) *
        100
      ).toFixed(0)}%`;
      break;
    }
    case "fileLoaded": {
      vm.scenarios = [
        ...[100, 200, 500, 1000].map((maxTrips) => {
          return {
            name: `Top ${maxTrips} trips with biggest spread (abs)`,
            msg: { op: "findInterestingTrips", attr: "maxSpread", maxTrips },
          };
        }),
        ...[100, 200, 500, 1000].map((maxTrips) => {
          return {
            name: `Top ${maxTrips} trips with biggest spread (rel)`,
            msg: { op: "findInterestingTrips", attr: "maxRelSpread", maxTrips },
          };
        }),
        ...[100, 150, 200, 250, 300].map((threshold) => {
          return {
            name: `Trips with max load >= ${threshold}%`,
            msg: {
              op: "findInterestingTrips",
              attr: "maxLoad",
              threshold: threshold / 100.0,
            },
          };
        }),
        {
          name: "Trips with P(over capacity) != 0% or 100%",
          msg: { op: "findInterestingTrips", attr: "uncertainOverCap" },
        },
        ...e.data.lines.map((line) => {
          return {
            name: dateTimeFormat.format(new Date(line.systemTime * 1000)),
            msg: { op: "getForecastInfo", line },
          };
        }),
      ];
      vm.loadingText = "";
      break;
    }
    case "getForecastInfoProgress": {
      vm.loadingText = `Loading forecast info... ${(
        (e.data.progress / e.data.size) *
        100
      ).toFixed(0)}%`;
      break;
    }
    case "tripForecast": {
      if (e.data.allEdgesHaveCapacity) {
        tripData.push(e.data);
      }
      break;
    }
    case "getForecastInfoDone": {
      vm.loadingText = "";
      const getTrainNr = (t) =>
        t.serviceInfos?.[0]?.train_nr || t.trip.train_nr || 0;
      tripData.sort((a, b) => getTrainNr(a) - getTrainNr(b));
      vm.trips = tripData.map((data) => {
        return {
          name: data.tripDisplayName,
          from: data.primaryStation?.name,
          to: data.secondaryStation?.name,
        };
      });
      break;
    }
    case "findInterestingTripsProgress": {
      vm.loadingText = `Finding interesting trips... ${(
        (e.data.progress / e.data.size) *
        100
      ).toFixed(0)}%`;
      break;
    }
    case "findInterestingTripsDone": {
      vm.loadingText = "";
      vm.trips = tripData.map((data) => {
        return {
          name:
            timeFormat.format(new Date(data.line.systemTime * 1000)) +
            ": " +
            data.tripDisplayName,
          from: data.primaryStation?.name,
          to: data.secondaryStation?.name,
        };
      });
      break;
    }
  }
});

window.addEventListener("load", () => {
  const dropZone = document;
  const preventDefault = (e) => {
    e.stopPropagation();
    e.preventDefault();
  };
  dropZone.addEventListener("dragenter", preventDefault);
  dropZone.addEventListener("dragover", preventDefault);
  dropZone.addEventListener("drop", (e) => {
    preventDefault(e);
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length !== 1) {
      alert("Multiple files are not supported");
      return;
    }
    vm.scenarios = [];
    vm.trips = [];
    vm.selectedScenario = "";
    vm.selectedTrip = "";
    vm.selectedTripData = null;
    vm.loadingText = "Loading file...";
    const file = files[0];
    worker.postMessage({ op: "loadFile", file: file });
  });
});
