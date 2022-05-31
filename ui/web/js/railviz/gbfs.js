var RailViz = RailViz || {};

/**
 * GBFS Bike Sharing / Vehicle Sharing symbol layers
 */
RailViz.GBFS = (function () {
    let map = null;
    let tilesEndpoint = null;
    let enabled = true;
    let gbfsOptions = new Map();
    let missedUpdate = null;

    function init(_map, _tilesEndpoint) {
        map = _map;
        tilesEndpoint = _tilesEndpoint;

        let loaded = 0;
        map.loadImage(`${tilesEndpoint}img/bike_marker.png`, (error, image) => {
            if (error) {
                throw error;
            }

            map.addImage("icon-bike", image);

            ++loaded;

            if (loaded == 2 && missedUpdate) {
                updateLayers(missedUpdate);
                missedUpdate = null;
            }
        });

        map.loadImage(`${tilesEndpoint}img/car_marker.png`, (error, image) => {
            if (error) {
                throw error;
            }

            map.addImage("icon-car", image);

            ++loaded;

            if (loaded == 2 && missedUpdate) {
                updateLayers(missedUpdate);
                missedUpdate = null;
            }
        });
    }

    function updateLayers(newOptionsList) {
        if (map === null) {
            missedUpdate = newOptionsList;
            return;
        }

        let add = [];
        let remove = [];

        newOptionsList.forEach(newOpt => {
            if (!gbfsOptions.has(newOpt.tag)) {
                add.push(newOpt);
            }
        });

        gbfsOptions.forEach((value, key) => {
            if (!newOptionsList.find(newOpt => newOpt.tag === key)) {
                remove.push(key);
            }
        })

        remove.forEach(tag => {
            console.log("removing GBFS layer", tag);
            gbfsOptions.delete(tag);
            map.removeLayer(`gbfs-${tag}-station`);
            map.removeLayer(`gbfs-${tag}-vehicle`);
            map.removeSource(`gbfs-${tag}`);
        });

        add.forEach(opt => {
            console.log("adding GBFS layer", opt.tag);
            gbfsOptions.set(opt.tag, opt);
            map.addSource(`gbfs-${opt.tag}`, {
                type: "vector",
                tiles: [`${tilesEndpoint}gbfs/tiles/${opt.tag}/{z}/{x}/{y}.mvt`],
                maxzoom: 20
            });
            map.addLayer({
                "id": `gbfs-${opt.tag}-vehicle`,
                "type": "symbol",
                "source": `gbfs-${opt.tag}`,
                "minzoom": 13,
                "source-layer": "vehicle",
                "layout": {
                    "icon-image": ['concat', 'icon-', ['get', 'type']],
                    "icon-size": ['interpolate', ['linear'], ['zoom'], 13, .6, 20, .8],
                    "icon-allow-overlap": false,
                    "text-allow-overlap": false
                },
            });
            map.addLayer({
                "id": `gbfs-${opt.tag}-station`,
                "type": "symbol",
                "source": `gbfs-${opt.tag}`,
                "minzoom": 13,
                "source-layer": "station",
                "layout": {
                    "text-field": ['case',
                      ['==', ['get', 'vehicles_available'], 999], '',
                      ["get", "vehicles_available"]
                    ],
                    "icon-image": ['concat', 'icon-', ['get', 'type']],
                    "text-font": ["Noto Sans Display Bold"],
                    "text-offset": [0.8, -1],
                    "text-size": 12,
                    "text-allow-overlap": false,
                    "icon-allow-overlap": false
                },
                "paint": {
                  "text-color": 'white',
                  "text-halo-color": ['case',
                    ['>=', ['get', 'vehicles_available'], 3], '#94BD46',
                    ['>=', ['get', 'vehicles_available'], 2], '#f8c11c',
                    ['==', ['get', 'vehicles_available'], 1], 'orange',
                    '#f83d11'
                  ],
                  "text-halo-width": 2
                }
            });
        });
    }

    function setEnabled(b) {
        if (map !== null) {
            ["railviz-base-line", "railviz-base-stations", "gbfs-stations"].forEach((l) => {
                if (map.getLayer(l) !== 'undefined') {
                    map.setLayoutProperty(l, "visibility", b ? "visible" : "none");
                }
            });
        }
        enabled = b;
    }

    return {
        init: init,
        setEnabled: setEnabled,
        updateLayers: updateLayers
    };
})();
