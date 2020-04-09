var RailViz = RailViz || {};

RailViz.Markers = (function () {

    var startMarker = null;
    var destinationMarker = null;
    var visible = true;
    var map;
    var startText = "";
    var destinationText = "";

    function init(mapRef) {
        map = mapRef;
    }

    function setLocale(l) {
        startText = l.start;
        destinationText = l.destination;

        if (startMarker) {
            startMarker.getElement().setAttribute("title", startText);
        }
        if (destinationMarker) {
            destinationMarker.getElement().setAttribute("title", destinationText);
        }
    }

    function setMarkers(settings) {
        if (startMarker) {
            if (settings.start) {
                startMarker.setLngLat([settings.start.lng, settings.start.lat]);
            } else if (visible) {
                startMarker.remove();
                startMarker = null;
            }
        } else {
            if (settings.start) {
                startMarker = new mapboxgl.Marker()
                    .setLngLat([settings.start.lng, settings.start.lat]);
                startMarker.getElement().setAttribute("title", startText);
                if (visible) {
                    startMarker.addTo(map);
                }
            }
        }

        if (destinationMarker) {
            if (settings.destination) {
                destinationMarker.setLngLat([settings.destination.lng, settings.destination.lat]);
            } else if (visible) {
                destinationMarker.remove();
                destinationMarker = null;
            }
        } else {
            if (settings.destination) {
                destinationMarker = new mapboxgl.Marker()
                    .setLngLat([settings.destination.lng, settings.destination.lat]);
                destinationMarker.getElement().setAttribute("title", destinationText);
                if (visible) {
                    destinationMarker.addTo(map);
                }
            }
        }
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
        hide: hide
    };

})();