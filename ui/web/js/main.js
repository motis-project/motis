"use strict";

(function bootstrap() {
  let params = getQueryParameters();
  let apiEndpoint = getApiEndpoint(params);

  const mapConfigUrl = apiEndpoint + "railviz/map_config";
  let mapConfigPromise = fetch(mapConfigUrl)
    .then((r) => r.json())
    .catch((e) => {
      console.error("could not fetch railviz/map_config", e);
      return { content: {} };
    });
  let windowLoadPromise = new Promise((resolve) =>
    window.addEventListener("load", resolve)
  );

  Promise.all([mapConfigPromise, windowLoadPromise]).then((promises) => {
    const mapConfig = promises[0].content;

    let tilesEndpoint = mapConfig.tiles_redirect;
    if (!tilesEndpoint) {
      tilesEndpoint = apiEndpoint;
      if (!tilesEndpoint.startsWith("http")) {
        tilesEndpoint = window.location.origin + tilesEndpoint;
      }
      if (!tilesEndpoint.endsWith("/")) {
        tilesEndpoint += "/";
      }
      tilesEndpoint += "tiles";
    }
    if (!tilesEndpoint.endsWith("/")) {
      tilesEndpoint += "/";
    }

    let initialPermalink = parseInitialPermalink(mapConfig.initial_permalink);

    let simulationTime = null;
    let timeParam = params["time"] || null;
    if (timeParam) {
      simulationTime = parseTimestamp(timeParam);
    }
    if (simulationTime == null) {
      timeParam = (initialPermalink.timestamp / 1000).toString();
      simulationTime = initialPermalink.timestamp;
    }

    let langParam = params["lang"] || null;
    let language = langParam || "de";

    window.app = Elm.Main.embed(document.getElementById("app-container"), {
      apiEndpoint: apiEndpoint,
      currentTime: Date.now(),
      simulationTime: simulationTime,
      language: language,
      motisParam: params["motis"] || null,
      timeParam: timeParam,
      langParam: langParam,
      fromLocation: localStorage.getItem("motis.routing.from_location"),
      toLocation: localStorage.getItem("motis.routing.to_location"),
      fromModes: localStorage.getItem("motis.routing.from_modes"),
      toModes: localStorage.getItem("motis.routing.to_modes"),
      intermodalPprMode: null,
    });

    window.elmMaps = {};

    initPorts(app, apiEndpoint, tilesEndpoint, initialPermalink);
    handleDrop(document.getElementById("app-container"));
    app.ports.localStorageSet.subscribe(function (kv) {
      localStorageSet(kv[0], kv[1]);
    });
  });
})();

function handleDrop(element) {
  element.addEventListener("drop", function (e) {
    e.preventDefault();
    var files = e.dataTransfer.files;
    if (files.length == 0) {
      return;
    }
    var contents = [];
    var remaining = files.length;

    function onLoad(i) {
      return function (ev) {
        contents[i] = ev.target.result;
        if (--remaining == 0) {
          var data = [];
          for (var j = 0; j < files.length; j++) {
            data[j] = [files[j].name, contents[j]];
          }
          app.ports.setRoutingResponses.send(data);
        }
      };
    }

    for (var i = 0; i < files.length; i++) {
      var reader = new FileReader();
      reader.addEventListener("load", onLoad(i));
      reader.readAsText(files[i]);
    }
  });
  element.addEventListener("dragenter", function (e) {
    e.preventDefault();
  });
  element.addEventListener("dragover", function (e) {
    e.preventDefault();
  });
}

function localStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (ex) {}
}

function getQueryParameters() {
  var params = {};
  window.location.search
    .substr(1)
    .split("&")
    .forEach((p) => {
      var param = p.split("=");
      params[param[0]] = decodeURIComponent(param[1]);
    });
  return params;
}

function getApiEndpoint(params) {
  let apiEndpoint = String(window.location.origin + window.location.pathname);
  let motisParam = params["motis"] || null;
  if (motisParam) {
    if (/^[0-9]+$/.test(motisParam)) {
      apiEndpoint = "https://" + defaultHost + ":" + motisParam;
    } else if (!motisParam.includes(":")) {
      apiEndpoint = "https://" + motisParam + ":" + defaultPort;
    } else if (
      !motisParam.startsWith("http:") &&
      !motisParam.startsWith("https:")
    ) {
      apiEndpoint = "https://" + motisParam;
    } else {
      apiEndpoint = motisParam;
    }
  }

  if (!apiEndpoint.endsWith("/")) {
    apiEndpoint += "/";
  }
  return apiEndpoint;
}

function parseTimestamp(value) {
  var filterInt = function (value) {
    if (/^(\-|\+)?([0-9]+|Infinity)$/.test(value)) return Number(value);
    return NaN;
  };
  if (value != null) {
    var time = filterInt(value);
    if (time) {
      return time * 1000;
    } else {
      var date = new Date(value);
      var time = date.getTime();
      return time && !isNaN(time) ? time : null;
    }
  }
  return null;
}

function parseInitialPermalink(str) {
  const parts = str ? str.split("/") : [];
  // fallback: Darmstadt
  return {
    lat: parseFloat(parts[1]) || 49.8728,
    lng: parseFloat(parts[2]) || 8.6512,
    zoom: parseInt(parts[3]) || 10,
    bearing: parseFloat(parts[4]) || 0,
    pitch: parseFloat(parts[5]) || 0,
    timestamp: parseTimestamp(parts[6]) || null,
  };
}
