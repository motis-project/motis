var RailViz = RailViz || {};

RailViz.Markers = (function () {
  let markerSettings = {};
  let startText = "";
  let destinationText = "";

  let startMarker = null;
  let destinationMarker = null;
  let visible = true;
  let map;

  function init(mapRef) {
    map = mapRef;
  }

  function setLocale(l) {
    startText = l.start;
    destinationText = l.destination;
    updateMarkers();

    if (startMarker) {
      startMarker.getElement().setAttribute("title", startText);
    }
    if (destinationMarker) {
      destinationMarker.getElement().setAttribute("title", destinationText);
    }
  }

  function setMarkers(settings) {
    markerSettings = settings;
    updateMarkers();
  }

  function updateMarkers() {
    startMarker = updateMarker(
      startMarker,
      markerSettings.startPosition,
      startText,
      markerSettings.startName,
      "url(img/marker_origin.png)",
      visible
    );

    destinationMarker = updateMarker(
      destinationMarker,
      markerSettings.destinationPosition,
      destinationText,
      markerSettings.destinationName,
      "url(img/marker_destination.png)",
      visible
    );
  }

  function updateMarker(m, pos, text, name, img, visible) {
    if (m) {
      if (!visible || !pos) {
        m.remove();
        m = null;
      } else {
        m.setLngLat([pos.lng, pos.lat]);
        m.getElement().setAttribute("title", `${text || ""}: ${name}`);
      }
    } else if (pos) {
      let el = document.createElement("div");
      el.style.display = "block";
      el.style.width = "32px";
      el.style.height = "48px";
      el.style.backgroundImage = img;

      m = new mapboxgl.Marker(el, {
        anchor: "bottom",
        offset: [0, 7],
      }).setLngLat([pos.lng, pos.lat]);
      m.getElement().setAttribute("title", `${text || ""}: ${name}`);
      if (visible) {
        m.addTo(map);
      }
    }
    return m;
  }

  function show() {
    if (!visible) {
      visible = true;
      if (startMarker) {
        startMarker.addTo(map);
      }
      if (destinationMarker) {
        destinationMarker.addTo(map);
      }
    }
  }

  function hide() {
    if (visible) {
      visible = false;
      if (startMarker) {
        startMarker.remove();
      }
      if (destinationMarker) {
        destinationMarker.remove();
      }
    }
  }

  return {
    init: init,
    setLocale: setLocale,
    setMarkers: setMarkers,
    show: show,
    hide: hide,
  };
})();
