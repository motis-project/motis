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
            startMarker.setTooltipContent(startText);
        }
        if (destinationMarker) {
            destinationMarker.setTooltipContent(destinationText);
        }
    }

    function setMarkers(settings) {
        if (startMarker) {
            if (settings.start) {
                startMarker.setLatLng(
                    L.latLng(settings.start.lat, settings.start.lng));
            } else if (visible) {
                startMarker.remove();
                startMarker = null;
            }
        } else {
            if (settings.start) {
                startMarker = L.marker(
                    L.latLng(settings.start.lat, settings.start.lng)).bindTooltip(startText);
                if (visible) {
                    startMarker.addTo(map);
                }
            }
        }

        if (destinationMarker) {
            if (settings.destination) {
                destinationMarker.setLatLng(
                    L.latLng(settings.destination.lat, settings.destination.lng));
            } else if (visible) {
                destinationMarker.remove();
                destinationMarker = null;
            }
        } else {
            if (settings.destination) {
                destinationMarker = L.marker(
                    L.latLng(settings.destination.lat, settings.destination.lng)).bindTooltip(destinationText);
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