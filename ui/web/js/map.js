// moveToMapPosition and syncMaps take from mapbox-gl-sync-move
// see : https://github.com/mapbox/mapbox-gl-sync-move
//
// Copyright (c) 2016, Mapbox
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function moveToMapPosition (master, clones) {
  var center = master.getCenter();
  var zoom = master.getZoom();
  var bearing = master.getBearing();
  var pitch = master.getPitch();

  clones.forEach(function (clone) {
    clone.jumpTo({
      center: center,
      zoom: zoom,
      bearing: bearing,
      pitch: pitch
    });
  });
}

// Sync movements of two maps.
//
// All interactions that result in movement end up firing
// a "move" event. The trick here, though, is to
// ensure that movements don't cycle from one map
// to the other and back again, because such a cycle
// - could cause an infinite loop
// - prematurely halts prolonged movements like
//   double-click zooming, box-zooming, and flying
function syncMaps () {
  var maps;
  var argLen = arguments.length;
  if (argLen === 1) {
    maps = arguments[0];
  } else {
    maps = [];
    for (var i = 0; i < argLen; i++) {
      maps.push(arguments[i]);
    }
  }

  // Create all the movement functions, because if they're created every time
  // they wouldn't be the same and couldn't be removed.
  var fns = [];
  maps.forEach(function (map, index) {
    fns[index] = sync.bind(null, map, maps.filter(function (o, i) { return i !== index; }));
  });

  function on () {
    maps.forEach(function (map, index) {
      map.on('move', fns[index]);
    });
  }

  function off () {
    maps.forEach(function (map, index) {
      map.off('move', fns[index]);
    });
  }

  // When one map moves, we turn off the movement listeners
  // on all the maps, move it, then turn the listeners on again
  function sync (master, clones) {
    off();
    moveToMapPosition(master, clones);
    on();
  }

  on();
  return function(){  off(); fns = []; };
}

// ---------------------------------------------------------------------------


class RailVizCustomLayer {
  constructor() {
    this.id =  'railviz_custom_layer';
    this.type =  'custom';
  }

  onAdd(map, gl) {
    this.map = map;
    RailViz.Render.setup(map, gl);

    const mouseEvents = ['mousedown', 'mouseup', 'mousemove', 'mouseout'];
    mouseEvents.forEach(
      eventType => map.getCanvas().addEventListener(
        eventType, event => RailViz.Render.handleMouseEvent(eventType, event)));

    // does not seem to work properly: https://github.com/mapbox/mapbox-gl-js/issues/9516
    map.on('webglcontextlost', () => {
      RailViz.Render.stop()
    });
    map.on('webglcontextrestored', () =>  {
      map.triggerRepaint();
      RailViz.Render.setup(map, gl);
    });

    this.updateViewportListener = () => this.updateViewport();
    map.on('moveend', this.updateViewportListener);

    this.updateViewport();
  }

  onRemove(map, gl) {
    map.off('moveend', this.updateViewportListener);
    RailViz.Render.stop();
  }

  prerender(gl, matrix) {
    RailViz.Render.prerender(gl, matrix, this.map.getZoom());
  }

  render(gl, matrix) {
    RailViz.Render.render(gl, matrix, this.map.getZoom());
  }

  updateViewport() {
    var rect = this.map.getCanvas().getBoundingClientRect();
    var center = this.map.getCenter();
    var zoom = Math.round(this.map.getZoom());
    var geoBounds = this.map.getBounds();
    // var railVizBounds = L.latLngBounds(
    //     this._map.unproject(pixelBounds.subtract(size)),
    //     this._map.unproject(pixelBounds.add(size).add(size)));
    var railVizBounds = geoBounds;

    var mapInfo = {
      scale: Math.pow(2, zoom),
      zoom: zoom,
      pixelBounds: {
        north: rect.top,
        west: rect.left,
        width: rect.width,
        height: rect.height
      },
      geoBounds: {
        north: geoBounds.getNorth(),
        west: geoBounds.getWest(),
        south: geoBounds.getSouth(),
        east: geoBounds.getEast()
      },
      railVizBounds: {
        north: railVizBounds.getNorth(),
        west: railVizBounds.getWest(),
        south: railVizBounds.getSouth(),
        east: railVizBounds.getEast()
      },
      center: {
        lat: center.lat,
        lng: center.lng
      }
    };

    app.ports.mapUpdate.send(mapInfo);
    RailViz.Main.mapUpdate(mapInfo);

    localStorageSet("motis.map", JSON.stringify({
      lat: center.lat,
      lng: center.lng,
      zoom: zoom
    }));
  }
};

function initPorts(app, apiEndpoint) {
  app.ports.mapInit.subscribe(function(id) {
    var mapSettings = localStorage.getItem("motis.map");
    if (mapSettings) {
      mapSettings = JSON.parse(mapSettings);
    }
    var lat = (mapSettings && mapSettings.lat) || 47.3684854; // 49.8728;
    var lng = (mapSettings && mapSettings.lng) || 8.5102601; // 8.6512;
    var zoom = (mapSettings && mapSettings.zoom) || 10;


    // use two maps until resolved: https://github.com/mapbox/mapbox-gl-js/issues/8159
    var map_bg = new mapboxgl.Map({
      container: `${id}-background`,
      style: {
        "version": 8,
        "sources": {
          "raster": {
            "type": "raster",
            "tiles": ["https://tiles.motis-project.de/osm_light/{z}/{x}/{y}.png?token=862bdec137edd4e88029304609458291f0ec760b668c5816ccdd83d0beae76a4"],
            "tileSize": 256,
            "maxzoom": 18
          }
        },
        "layers": [{
            "id": "raster",
            "type": "raster",
            "source": "raster"
        }]
      },
      zoom: zoom,
      center: [lng, lat],
      antialias: true
    });

    const empty = {
      version: 8,
      name: "Empty",
      metadata: {
        "mapbox:autocomposite": true
      },
      sources: {
        "vectors": {
          "type": "vector",
          "tiles": [`http://han.algo.informatik.tu-darmstadt.de:8082/path/tiles/{z}/{x}/{y}.mvt`],
          "maxzoom": 20
        }
      },
      layers: [
        {
          "id": "path",
          "type": "line",
          "source": "vectors",
          "source-layer": "path",
          "layout": {
            "line-join": "round",
            "line-cap": "round"
          },
          "paint": {
            "line-color": "hsl(0, 0, 30%)",
            "line-width": 4
          }
        }, {
          "id": "station",
          "type": "circle",
          "source": "vectors",
          "source-layer": "station",
          "paint": {
            "circle-color": "white",
            "circle-radius": 2.5,
            "circle-stroke-color": "black",
            "circle-stroke-width": 2

          }
        }
      ]
    };

    var map_fg = new mapboxgl.Map({
      container: `${id}-foreground`,
      zoom: zoom,
      center: [lng, lat],
      style: empty,
      antialias: true
    });
    window.elmMaps[id] = map_fg;

    syncMaps(map_bg, map_fg);

    map_fg.on('load', () => {
      map_fg.addLayer(new RailVizCustomLayer());
    });

    map_fg.on('dragend', () => RailViz.Main.dragEnd());

    map_fg.on('contextmenu', e => {
      console.log("Context menu:", e);
      app.ports.mapShowContextMenu.send({
        mouseX: Math.round(e.point.x),
        mouseY: Math.round(e.point.y),
        lat: e.lngLat.lat,
        lng: e.lngLat.lng
      });
    });

    const padding = {top: 36, right: 24, bottom: 16, left: 624 };

    app.ports.mapFlyTo.subscribe(function(opt) {
      var map = window.elmMaps[opt.mapId];

      const options = map.cameraForBounds(
        new mapboxgl.LngLatBounds([opt.lng, opt.lat], [opt.lng, opt.lat]),
        { maxZoom : 12, padding }
      );
      if(opt.zoom) {
        options.zoom = opt.zoom;
      }

      if(opt.animate) {
        map.flyTo(options);
      } else {
        map.jumpTo(options);
      }
    });

    app.ports.mapFitBounds.subscribe(function(opt) {
      const bounds = opt.coords.reduce((b, c) => b.extend([c[1], c[0]]),
                                       new mapboxgl.LngLatBounds());
      if(!bounds.isEmpty()) {
        window.elmMaps[opt.mapId]
              .fitBounds(bounds, { padding });
      }
    });

    app.ports.mapSetConnections.subscribe(function(opt) {
      const bounds = opt.connections
                        .reduce((b, conn) => conn.stations
                                                 .reduce((sb, s) => sb.extend([s.pos.lng, s.pos.lat]),
                                                         b),
                                new mapboxgl.LngLatBounds());
      if(!bounds.isEmpty()) {
        window.elmMaps[opt.mapId]
              .fitBounds(bounds, { padding });
      }
    });

    RailViz.Main.init(apiEndpoint, app.ports);

    RailViz.Markers.init(map_fg);
    app.ports.mapSetMarkers.subscribe(RailViz.Markers.setMarkers);

    app.ports.setRailVizFilter.subscribe(RailViz.Main.setTripFilter);
    app.ports.mapSetConnectionFilter.subscribe(RailViz.Main.setConnectionFilter);
    app.ports.mapSetConnections.subscribe(RailViz.Main.setConnections);
    app.ports.mapHighlightConnections
            .subscribe(RailViz.Main.highlightConnections);
    app.ports.mapUpdateWalks.subscribe(RailViz.Main.updateWalks);
    app.ports.setTimeOffset.subscribe(RailViz.Main.setTimeOffset);
    app.ports.setPPRSearchOptions.subscribe(RailViz.Main.setPPRSearchOptions);

    app.ports.mapUseTrainClassColors.subscribe(
      RailViz.Trains.setUseCategoryColor
    );
    app.ports.mapShowTrains.subscribe(RailViz.Main.showTrains);
    app.ports.mapSetLocale.subscribe(RailViz.Markers.setLocale);
  });
}
