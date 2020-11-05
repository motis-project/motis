const worker = new Worker("worker.js");

const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "long",
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
    svgGraphWidth() {
      return this.tripSectionCount * 50;
    },
    svgMidpointX() {
      return this.svgGraphWidth / 2;
    },
    svgViewBox() {
      return `-30 -15 ${50 + this.svgGraphWidth} 235`;
    },
    svgDividers() {
      if (!this.selectedTripData) {
        return [];
      }
      return this.selectedTripData.edges.map(
        (_, idx) => `M${(idx + 1) * 50} 0v200`
      );
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
    svgProbRegions() {
      let r = {
        maxLoad: 100,
        p100: 50,
        p80: 80,
      };
      if (this.selectedTripData) {
        r.maxLoad = Math.max(1.1, this.selectedTripData.maxLoad + 0.1);
        r.p100 = 200 - Math.round(200 / r.maxLoad);
        r.p80 = 200 - Math.round((0.8 * 200) / r.maxLoad);
      }
      return r;
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
        const maxLoad = this.svgProbRegions.maxLoad;
        let x = 0;
        for (const ef of this.selectedTripData.edges) {
          const topLoad = ef.q_80 / ef.capacity;
          const bottomLoad = ef.q_20 / ef.capacity;
          const topY = 200 - Math.round((topLoad / maxLoad) * 200);
          const bottomY = 200 - Math.round((bottomLoad / maxLoad) * 200);
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
    svgStations() {
      if (!this.selectedTripData) {
        return [];
      }
      return [this.selectedTripData.edges[0].from]
        .concat(this.selectedTripData.edges.map((ef) => ef.to))
        .map((station, idx) => {
          return {
            x: idx * 50,
            eva: station.id,
            name: station.name,
          };
        });
    },
  },
  methods: {
    formatDateTime(timestamp) {
      return dateTimeFormat.format(new Date(timestamp * 1000));
    },
  },
  watch: {
    selectedScenario(newScenario) {
      const scenario = this.scenarios[parseInt(newScenario)];
      tripData = [];
      this.trips = [];
      this.selectedTrip = "";
      vm.loadingText = "Loading forecast info...";
      worker.postMessage({ op: "getForecastInfo", line: Vue.toRaw(scenario) });
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
    const maxLoad = vm.svgProbRegions.maxLoad;
    let x = 0;
    for (const ef of td.edges) {
      const load = (ef[prop] || 0) / ef.capacity;
      const y = 200 - Math.round((load / maxLoad) * 200);
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
      vm.scenarios = e.data.lines;
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
