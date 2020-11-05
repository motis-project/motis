importScripts("lib/clarinet.js", "json_stream.js");

let loadedFile;

async function* lineIterator(file, maxPrefixLen = 1024) {
  const stream = file.stream();
  const reader = stream.getReader();
  const progressUpdateStep = 100 * 1024 * 1024;

  try {
    let lineStartOffset = 0;
    let { value: chunk, done: readerDone } = await reader.read();
    if (readerDone) {
      return;
    }
    let nextFileOffset = chunk.length;
    let lastProgressOffset = 0;

    const linePrefix = new Uint8Array(maxPrefixLen);
    let linePrefixLen = Math.min(chunk.length, linePrefix.length);
    linePrefix.set(chunk.subarray(0, linePrefixLen));

    const makeLineInfo = (lineEndOffset) => {
      return {
        begin: lineStartOffset,
        end: lineEndOffset,
        length: lineEndOffset - lineStartOffset + 1,
        linePrefix: linePrefix.subarray(0, linePrefixLen),
      };
    };

    for (;;) {
      if (nextFileOffset - lastProgressOffset >= progressUpdateStep) {
        postMessage({
          op: "fileLoadProgress",
          offset: nextFileOffset,
          size: file.size,
        });
        lastProgressOffset = nextFileOffset;
      }
      const newLineIdx = chunk.indexOf(10);
      if (newLineIdx === -1) {
        ({ value: chunk, done: readerDone } = await reader.read());
        if (readerDone) {
          break;
        }
        nextFileOffset += chunk.length;
        if (linePrefixLen < linePrefix.length) {
          const copyLen = Math.min(
            chunk.length,
            linePrefix.length - linePrefixLen
          );
          linePrefix.set(chunk.subarray(0, copyLen), linePrefixLen);
          linePrefixLen += copyLen;
        }
        continue;
      }
      const lineEndOffset = nextFileOffset - chunk.length + newLineIdx;
      yield makeLineInfo(lineEndOffset);
      chunk = chunk.subarray(newLineIdx + 1);
      lineStartOffset = lineEndOffset + 1;
      linePrefixLen = Math.min(chunk.length, linePrefix.length);
      linePrefix.set(chunk.subarray(0, linePrefixLen));
    }
    if (chunk && chunk.length !== 0) {
      yield makeLineInfo(nextFileOffset);
    }
  } finally {
    reader.releaseLock();
  }
}

function extractSystemTime(linePrefix) {
  const decoder = new TextDecoder("utf-8");
  const startStr = decoder.decode(linePrefix);

  let systemTime;
  let nextValueIsTime = false;

  const parser = clarinet.parser();

  parser.onvalue = (val) => {
    if (nextValueIsTime) {
      console.assert(
        typeof val === "number",
        "system_time is not a number: %s",
        val
      );
      systemTime = val;
      nextValueIsTime = false;
    }
  };

  parser.onkey = (key) => {
    if (key === "system_time") {
      nextValueIsTime = true;
    }
  };

  parser.onopenobject = parser.onkey;

  parser.write(startStr).close();

  return systemTime;
}

async function loadFile(file) {
  loadedFile = file;
  let lines = [];

  for await (const line of lineIterator(file)) {
    line.systemTime = extractSystemTime(line.linePrefix);
    delete line.linePrefix;
    lines.push(line);
  }

  postMessage({ op: "fileLoaded", file, lines });
}

function probPaxLE(cdf, limit) {
  let prob = 0.0;
  for (const e of cdf) {
    if (e.passengers > limit) {
      break;
    }
    prob = e.probability;
  }
  return prob;
}

function probPaxGT(cdf, limit) {
  return 1.0 - probPaxLE(cdf, limit);
}

function paxQuantile(cdf, q) {
  let last = null;
  for (const e of cdf) {
    if (e.probability == q) {
      return e.passengers;
    } else if (e.probability > q) {
      if (last !== null) {
        return (last.passengers + e.passengers) / 2;
      } else {
        return e.passengers;
      }
    }
    last = e;
  }
  throw "invalid cdf";
}

function processEdgeForecast(ef) {
  if (!ef.capacity) {
    return ef;
  }
  ef.p_load_gt_100 = probPaxGT(ef.passenger_cdf, ef.capacity);
  ef.q_20 = paxQuantile(ef.passenger_cdf, 0.2);
  ef.q_50 = paxQuantile(ef.passenger_cdf, 0.5);
  ef.q_80 = paxQuantile(ef.passenger_cdf, 0.8);
  ef.max_pax =
    ef.passenger_cdf.length > 0
      ? ef.passenger_cdf[ef.passenger_cdf.length - 1].passengers
      : 0;
  return ef;
}

function getTripDisplayName(trip, serviceInfos) {
  if (serviceInfos && serviceInfos.length > 0) {
    return serviceInfos
      .map((si) => (si.line ? `${si.name} [${si.train_nr}]` : si.name))
      .join(", ");
  } else {
    return (trip.train_nr || 0).toString();
  }
}

async function getForecastInfo(file, line) {
  const fileSlice = file.slice(line.begin, line.end, "application/json");
  const stream = new JSONStream();
  const progressUpdateStep = 64 * 1024;
  let lastProgressUpdate = 0;

  stream.onprogress = (progress, size) => {
    if (progress - lastProgressUpdate >= progressUpdateStep) {
      postMessage({
        op: "getForecastInfoProgress",
        file,
        line,
        progress,
        size,
      });
      lastProgressUpdate = progress;
    }
  };

  stream.onkey = (key) => {
    const depth = stream.currentDepth();
    return (
      depth === 4 &&
      (key === "trip" ||
        key === "edges" ||
        key === "primary_station" ||
        key === "secondary_station" ||
        key === "service_infos")
    );
  };

  let currentTrip = null;
  let primaryStation = null;
  let secondaryStation = null;
  let serviceInfos = null;

  stream.onobject = (key, obj) => {
    switch (key) {
      case "trip":
        currentTrip = obj;
        break;
      case "primary_station":
        primaryStation = obj;
        break;
      case "secondary_station":
        secondaryStation = obj;
        break;
      default:
        console.log("unexpected object:", key, obj);
    }
  };

  stream.onarray = (key, arr) => {
    if (key === "service_infos") {
      serviceInfos = arr;
      return;
    }
    console.assert(key === "edges", "unexpected array:", key, arr);
    console.assert(currentTrip, "missing trip");
    const edges = arr
      .filter((ef) => ef.from.id != ef.to.id)
      .map(processEdgeForecast);
    const allEdgesHaveCapacity = arr.every((ef) => ef.capacity);
    const maxLoad = edges.reduce(
      (max, ef) =>
        ef.capacity ? Math.max(max, ef.max_pax / ef.capacity) : max,
      0.0
    );
    postMessage({
      op: "tripForecast",
      file,
      line,
      trip: currentTrip,
      edges,
      allEdgesHaveCapacity,
      maxLoad,
      primaryStation,
      secondaryStation,
      serviceInfos,
      tripDisplayName: getTripDisplayName(currentTrip, serviceInfos),
    });
    currentTrip = null;
  };

  await stream.parseBlob(fileSlice);
  postMessage({ op: "getForecastInfoDone", file, line });
}

addEventListener("message", (e) => {
  switch (e.data.op) {
    case "loadFile":
      return loadFile(e.data.file);
    case "getForecastInfo":
      return getForecastInfo(loadedFile, e.data.line);
  }
});
