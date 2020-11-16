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
        const y100 = 200 - Math.round((e.capacity / maxVal) * 200);
        const y80 = 200 - Math.round(((e.capacity * 0.8) / maxVal) * 200);
        return {
          capacity: e.capacity,
          x: idx * 50,
          yRed: 0,
          hRed: y100,
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
      const stepSize = maxVal >= 700 ? 100 : 50;
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
  },
  methods: {
    formatDateTime(timestamp) {
      return dateTimeFormat.format(new Date(timestamp * 1000));
    },
    formatTime(timestamp) {
      return timestamp ? timeFormat.format(new Date(timestamp * 1000)) : "";
    },
  },
  watch: {
    selectedScenario(newScenario) {
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
    const file = files[0];
    worker.postMessage({ op: "loadFile", file: file });
  });
});
