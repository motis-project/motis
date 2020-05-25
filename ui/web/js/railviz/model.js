var RailViz = RailViz || {};

RailViz.Model = (function () {
  class PolylineRef {
    constructor(fwd, offset, polyline) {
      this.fwd = fwd;
      this.offset = offset;
      this.polyline = polyline;
    }
  }

  class Train {
    constructor(t, data) {
      Object.assign(this, t);

      this.clasz = this.clasz || 0;
      this.trip.forEach((trp) => (trp.train_nr = trp.train_nr || 0));

      const findStationById = (id) => {
        return data.stations.find((s) => s.id == id);
      };
      this.dStation = findStationById(this.d_station_id);
      this.aStation = findStationById(this.a_station_id);

      let offset = 0;
      this.polylines = this.indices.map((i) => {
        const pr = new PolylineRef(i > 0, offset, data.polylines[Math.abs(i)]);
        offset += pr.polyline.totalLength;
        return pr;
      });
      this.totalLength = offset;

      // index into this.polylines
      this.currPolyline = null;
      // index into this.polylines[this.currPolyline]
      this.currLine = null;
      // [0-1] in  this.polylines[this.currPolyline].foo[this.currLine]
      this.currProgress = null;
    }

    firstCoordinate() {
      const p = this.polylines[0];
      const coords = p.polyline.coordinates;
      return coords[p.fwd ? 0 : coords.length - 1];
    }

    lastCoordinate() {
      const p = this.polylines[this.polylines.length - 1];
      const coords = p.polyline.coordinates;
      return coords[p.fwd ? coords.length - 1 : 0];
    }

    firstMercatorCoordinate() {
      const p = this.polylines[0];
      const coords = p.polyline.mercatorCoordinates;
      return coords[p.fwd ? 0 : coords.length - 1];
    }

    lastMercatorCoordinate() {
      const p = this.polylines[this.polylines.length - 1];
      const coords = p.polyline.mercatorCoordinates;
      return coords[p.fwd ? coords.length - 1 : 0];
    }

    updatePosition(time) {
      if (time < this.d_time || time > this.a_time) {
        // not visible at all
        const update = this.currPolyline != null;
        this.currPolyline = null;
        this.currLine = null;
        this.currProgress = null;
        return update;
      }

      const progress = (time - this.d_time) / (this.a_time - this.d_time);
      const pos = progress * this.totalLength;

      // was (and is) visible; same line (and same polyline)
      if (this.currPolyline != null) {
        const p = this.polylines[this.currPolyline];
        const [lOffset, lLength] = this.getCurrLinePos(p);
        if (pos >= lOffset && pos <= lOffset + lLength) {
          this.currProgress = (pos - lOffset) / lLength || 0;
          return false;
        }
      }

      this.updateCurrPolyline(pos);
      this.updateCurrLine(pos);

      const p = this.polylines[this.currPolyline];
      const [lOffset, lLength] = this.getCurrLinePos(p);
      this.currProgress = (pos - lOffset) / lLength || 0;

      return true;
    }

    updateCurrPolyline(pos) {
      this.currPolyline = this.currPolyline || 0;
      const oldPolyline = this.currPolyline;
      for (; this.currPolyline < this.polylines.length; ++this.currPolyline) {
        const p = this.polylines[this.currPolyline];
        if (pos >= p.offset && pos <= p.offset + p.polyline.totalLength) {
          if (this.currPolyline != oldPolyline) {
            this.currLine = 0;
          }
          break;
        }
      }
      this.currPolyline = Math.min(
        this.currPolyline,
        this.polylines.length - 1
      );
    }

    updateCurrLine(pos) {
      this.currLine = this.currLine || 0;
      const p = this.polylines[this.currPolyline];
      for (; this.currLine < p.polyline.lineCount; ++this.currLine) {
        const [lOffset, lLength] = this.getCurrLinePos(p);
        if (pos >= lOffset && pos <= lOffset + lLength) {
          break;
        }
      }
      this.currLine = Math.min(this.currLine, p.polyline.lineCount - 1);
    }

    getCurrLinePos(p /* :PolylineRef */) {
      if (p.fwd) {
        return [
          p.offset + p.polyline.lineOffsets[this.currLine],
          p.polyline.lineLengths[this.currLine],
        ];
      } else {
        const idx = p.polyline.lineCount - 1 - this.currLine;
        const offset = p.polyline.lineOffsets[idx];
        const length = p.polyline.lineLengths[idx];
        return [p.offset + p.polyline.totalLength - (offset + length), length];
      }
    }

    getMercatorLine() {
      if (this.currPolyline == null) {
        return null;
      }

      const p = this.polylines[this.currPolyline];
      const mc = p.polyline.mercatorCoordinates;
      if (p.fwd) {
        const offset = this.currLine;
        return [mc[offset].x, mc[offset].y, mc[offset + 1].x, mc[offset + 1].y];
      } else {
        const offset = p.polyline.lineCount - this.currLine;
        return [mc[offset].x, mc[offset].y, mc[offset - 1].x, mc[offset - 1].y];
      }
    }
  }

  function preprocess(data) {
    data.stations.forEach((s) => preprocessStation(s));
    data.polylines = data.polylines.map(preprocessPolyline);
    data.trains = data.trains.map((t) => new Train(t, data));

    data.trains.sort((lhs, rhs) =>
      rhs.clasz != lhs.clasz
        ? rhs.clasz - lhs.clasz
        : (rhs.route_distance_ || 0) - (lhs.route_distance_ || 0)
    );
  }

  function preprocessStation(s) {
    s.rawPos = s.pos;
    s.mercatorPos = mapboxgl.MercatorCoordinate.fromLngLat([
      s.pos.lng,
      s.pos.lat,
    ]);
  }

  function preprocessPolyline(str) {
    const coordinates = polyline.decode(str, 6);
    if (coordinates.length == 0) {
      return {};
    }

    let p = {
      coordinates: coordinates,
      mercatorCoordinates: coordinates.map((ll) =>
        mapboxgl.MercatorCoordinate.fromLngLat([ll[1], ll[0]])
      ),
    };

    let mc = p.mercatorCoordinates;
    p.lineCount = mc.length - 1;
    p.lineLengths = new Array(p.lineCount);
    p.lineOffsets = new Array(p.lineCount + 1);

    let offset = 0;
    for (let i = 0; i < p.lineCount; i++) {
      const x_dist = mc[i].x - mc[i + 1].x;
      const y_dist = mc[i].y - mc[i + 1].y;
      const length = Math.sqrt(x_dist * x_dist + y_dist * y_dist);
      p.lineLengths[i] = length;
      p.lineOffsets[i] = offset;
      offset += length;
    }
    p.lineOffsets[p.lineCount] = offset;
    p.totalLength = offset;

    return p;
  }

  function convertPolyline(polyline) {
    const coords = polyline.coordinates;
    let converted = [];
    let j = 0;
    for (let i = 0; i < coords.length - 1; i += 2) {
      const wc = mapboxgl.MercatorCoordinate.fromLngLat({
        lng: coords[i + 1],
        lat: coords[i],
      });
      if (j == 0 || wc.x != converted[j - 2] || wc.y != converted[j - 1]) {
        converted[j] = wc.x;
        converted[j + 1] = wc.y;
        j += 2;
      }
    }
    polyline.coordinates = converted;
  }

  function prepareWalk(walk) {
    preprocessStation(walk.departureStation);
    preprocessStation(walk.arrivalStation);
    walk.from_station_id = walk.departureStation.id;
    walk.to_station_id = walk.arrivalStation.id;
    if (walk.polyline) {
      walk.coordinates = {
        coordinates: walk.polyline,
      };
      convertPolyline(walk.coordinates);
    } else {
      walk.coordinates = {
        coordinates: [
          walk.departureStation.pos.x,
          walk.departureStation.pos.y,
          walk.arrivalStation.pos.x,
          walk.arrivalStation.pos.y,
        ],
      };
    }
  }

  function doublesToLngLats(src) {
    let converted = [];
    for (let i = 0; i < src.length - 1; i += 2) {
      const ll = [src[i + 1], src[i]];
      if (
        converted.length == 0 ||
        ll[0] != converted[converted.length - 1][0] ||
        ll[1] != converted[converted.length - 1][1]
      ) {
        converted.push(ll);
      }
    }
    return converted;
  }

  return {
    preprocess: preprocess,
    convertPolyline: convertPolyline,
    prepareWalk: prepareWalk,
    doublesToLngLats: doublesToLngLats,
  };
})();
