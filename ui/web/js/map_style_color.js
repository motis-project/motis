var colorStyle = function (tilesEndpoint, apiEndpoint) {
    const matchLevel = (expression) => {
        return ["all",
            ["any",
                ["!has", "level"],
                ["==", ["get", "level"], level]
            ],
            expression
        ];
    }

    const level = 0;

    const water = "#99ddff";
    const rail = "#a8a8a8";
    const pedestrian = "#e8e7eb";

    const sport = "#d0f4be";
    const sport_outline = "#b3e998";

    const building = "#ded7d3";
    const building_outline = "#cfc8c4";

    return {
        "version": 8,
        "sources": {
            "osm": {
                "type": "vector",
                "tiles": [`${tilesEndpoint}{z}/{x}/{y}.mvt`],
                "maxzoom": 20
            }
        },
        "glyphs": `${tilesEndpoint}glyphs/{fontstack}/{range}.pbf`,
        "sprite": `${apiEndpoint}sprite/sprite`,
        "layers": [
            {
                "id": "background",
                "type": "background",
                "paint": {"background-color": "#f8f4f0"}
            }, {
                "id": "coastline",
                "type": "fill",
                "source": "osm",
                "source-layer": "coastline",
                "paint": {"fill-color": water}
            },
            {
                "id": "landuse_park",
                "type": "fill",
                "source": "osm",
                "source-layer": "landuse",
                "filter": ["==", ["get", "landuse"], "park"],
                "paint": {
                    "fill-color": "#b8ebad",
                    // "fill-outline-color": "rgba(95, 208, 100, 1)"
                }
            },
            {
                "id": "landuse",
                "type": "fill",
                "source": "osm",
                "source-layer": "landuse",
                "filter": [
                    "!in",
                    "landuse",
                    "park",
                    "public_transport"
                ],
                "paint": {
                    "fill-color": ["match", ["get", "landuse"],
                        "complex", "#f0e6d1",
                        "commercial", "hsla(0, 60%, 87%, 0.23)",
                        "industrial", "#e0e2eb",
                        "residential", "#ece7e4",
                        "retail", "hsla(0, 60%, 87%, 0.23)",
                        "construction", "#aaa69d",

                        "nature_light", "#ddecd7",
                        "nature_heavy", "#a3e39c",
                        "cemetery", "#e0e4dd",
                        "beach", "#fffcd3",

                        "magenta"]
                }
            },
            {
                "id": "water",
                "type": "fill",
                "source": "osm",
                "source-layer": "water",
                "paint": {"fill-color": water}
            },
            {
                "id": "sport",
                "type": "fill",
                "source": "osm",
                "source-layer": "sport",
                "paint": {
                    "fill-color": sport,
                    "fill-outline-color": sport_outline
                }
            },
            {
                "id": "pedestrian",
                "type": "fill",
                "source": "osm",
                "source-layer": "pedestrian",
                "paint": {"fill-color": pedestrian}
            },
            {
                "id": "waterway",
                "type": "line",
                "source": "osm",
                "source-layer": "waterway",
                "paint": {"line-color": water}
            },
            {
                "id": "building",
                "type": "fill",
                "source": "osm",
                "source-layer": "building",
                "paint": {
                    "fill-color": building,
                    "fill-outline-color": building_outline,
                    "fill-opacity":
                        ["interpolate", ["linear"], ["zoom"],
                            14, 0,
                            16, 0.8,
                        ]
                }
            },
            {
                "id": "indoor-corridor",
                "type": "fill",
                "source": "osm",
                "source-layer": "indoor",
                "filter": [
                    "all",
                    ["==", "indoor", "corridor"],
                    ["==", "level", level]
                ],
                "paint": {
                    "fill-color": "#fdfcfa",
                    "fill-opacity": 0.8
                }
            },
            {
                "id": "indoor",
                "type": "fill",
                "source": "osm",
                "source-layer": "indoor",
                "filter": [
                    "all",
                    ["!in", "indoor", "corridor", "wall", "elevator"],
                    ["==", "level", level]
                ],
                "paint": {
                    "fill-color": "#d4edff",
                    "fill-opacity": 0.8
                }
            },
            {
                "id": "indoor-outline",
                "type": "line",
                "source": "osm",
                "source-layer": "indoor",
                "filter": [
                    "all",
                    ["!in", "indoor", "corridor", "wall", "elevator"],
                    ["==", "level", level]
                ],
                "minzoom": 18,
                "paint": {
                    "line-color": "#808080",
                    "line-width": 2
                }
            },
            {
                "id": "indoor-names",
                "type": "symbol",
                "source": "osm",
                "source-layer": "indoor",
                "minzoom": 18,
                "filter": ["any", ["!has", "level"], ["==", "level", level]],
                "layout": {
                    "symbol-placement": "point",
                    "text-field": ["get", "name"],
                    "text-font": ["Noto Sans Display Regular"],
                    "text-size": 12,
                },
                "paint": {
                    "text-color": "#333333"
                }
            },
            {
                "id": "landuse-public-transport",
                "type": "fill",
                "source": "osm",
                "source-layer": "landuse",
                "filter": [
                    "all",
                    ["==", "landuse", "public_transport"],
                    ["any", ["!has", "level"], ["==", "level", level]]
                ],
                "paint": {
                    "fill-color": "rgba(218,140,140,0.3)"
                }
            },
            {
                "id": "footway",
                "type": "line",
                "source": "osm",
                "source-layer": "road",
                "filter": [
                    "all",
                    [
                        "in",
                        "highway",
                        "footway", "track", "cycleway", "path", "unclassified", "service"
                    ],
                    level === 0
                        ?
                        ["any",
                            ["!has", "level"],
                            ["==", "level", level]
                        ]
                        :
                        ["==", "level", level]
                ],
                "layout": {
                    "line-cap": "round",
                },
                "minzoom": 14,
                "paint": {
                    "line-dasharray": [ 0.75, 1.5 ],
                    "line-color": "#fff",
                    "line-opacity": 0.5,
                    "line-width": [
                        "let",
                        "base", 0.4,
                        ["interpolate", ["linear"], ["zoom"],
                            5, ["+", ["*", ["var", "base"], 0.1], 1],
                            9, ["+", ["*", ["var", "base"], 0.4], 1],
                            12, ["+", ["*", ["var", "base"], 1], 1],
                            16, ["+", ["*", ["var", "base"], 4], 1],
                            20, ["+", ["*", ["var", "base"], 8], 1]
                        ]
                    ]
                }
            },
            {
                "id": "steps",
                "type": "line",
                "source": "osm",
                "source-layer": "road",
                "minzoom": 18,
                "filter": [
                    "all",
                    ["==", "highway", "steps"],
                    level === 0
                        ?
                        ["any",
                            ["!has", "from_level"],
                            ["any",
                                ["==", "from_level", level],
                                ["==", "to_level", level]
                            ]
                        ]
                        :
                        ["any",
                            ["==", "from_level", level],
                            ["==", "to_level", level]
                        ]
                ],
                "paint": {
                    "line-dasharray": [ 0.5, 0.5 ],
                    "line-color": "#ff4524",
                    "line-opacity": 1,
                    "line-width": [
                        "let",
                        "base", 0.4,
                        ["interpolate", ["linear"], ["zoom"],
                            5, ["+", ["*", ["var", "base"], 0.1], 1],
                            9, ["+", ["*", ["var", "base"], 0.4], 1],
                            12, ["+", ["*", ["var", "base"], 1], 1],
                            16, ["+", ["*", ["var", "base"], 4], 1],
                            20, ["+", ["*", ["var", "base"], 8], 1]
                        ]
                    ]
                }
            },
            {
                "id": "indoor-elevator-outline",
                "type": "circle",
                "source": "osm",
                "source-layer": "indoor",
                "minzoom": 18,
                "filter": [
                    "all",
                    ["==", "indoor", "elevator"],
                    ["<=", "from_level", level],
                    [">=", "to_level", level]
                ],
                "paint": {
                    'circle-color': "#808080",
                    'circle-radius': 16,
                }
            },
            {
                "id": "indoor-elevator",
                "type": "circle",
                "source": "osm",
                "source-layer": "indoor",
                "minzoom": 18,
                "filter": [
                    "all",
                    ["==", "indoor", "elevator"],
                    ["<=", "from_level", level],
                    [">=", "to_level", level]
                ],
                "paint": {
                    'circle-color': '#bcf1ba',
                    'circle-radius': 14,
                }
            },
            {
                "id": "indoor-elevator-icon",
                "type": "symbol",
                "source": "osm",
                "source-layer": "indoor",
                "minzoom": 18,
                "filter": [
                    "all",
                    ["==", "indoor", "elevator"],
                    ["<=", "from_level", level],
                    [">=", "to_level", level]
                ],
                "layout": {
                    "icon-image": "elevator",
                    "icon-size": 0.9
                }
            },
            {
                "id": "road_back_residential",
                "type": "line",
                "source": "osm",
                "source-layer": "road",
                "filter": [ "==", "highway", "residential" ],
                "layout": {
                    "line-cap": "round",
                },
                "paint": {
                    "line-color": "#ffffff",
                    "line-width": ["interpolate", ["linear"], ["zoom"],
                        5, 0,
                        9, 0.5,
                        12, 1,
                        16, 4,
                        20, 20
                    ],
                    "line-opacity": ["interpolate", ["linear"], ["zoom"],
                        12, 0.4,
                        15, 1
                    ]
                }
            },
            {
                "id": "road_back_non_residential",
                "type": "line",
                "source": "osm",
                "source-layer": "road",
                "filter": [
                    "!in",
                    "highway",
                    "footway", "track", "steps", "cycleway", "path", "unclassified", "residential", "service"
                ],
                "layout": {
                    "line-cap": "round",
                },
                "paint": {
                    "line-color": "#ffffff",
                    "line-width": [
                        "let",
                        "base", ["match", ["get", "highway"],
                            "motorway", 4,
                            ["trunk", "motorway_link"], 3.5,
                            ["primary", "secondary", "aeroway", "trunk_link"], 3,
                            ["primary_link", "secondary_link", "tertiary", "tertiary_link"], 1.75,
                            0.0],
                        ["interpolate", ["linear"], ["zoom"],
                            5, ["+", ["*", ["var", "base"], 0.1], 1],
                            9, ["+", ["*", ["var", "base"], 0.4], 1],
                            12, ["+", ["*", ["var", "base"], 1], 1],
                            16, ["+", ["*", ["var", "base"], 4], 1],
                            20, ["+", ["*", ["var", "base"], 8], 1]
                        ]
                    ]
                }
            },
            {
                "id": "road",
                "type": "line",
                "source": "osm",
                "source-layer": "road",
                "layout": {
                    "line-cap": "round",
                },
                "filter": ["all",
                    ["has", "ref"],
                    ["any", ["==", ["get", "highway"], "motorway"],
                        ["==", ["get", "highway"], "trunk"],
                        ["==", ["get", "highway"], "secondary"],
                        [">", ["zoom"], 11]]],
                "paint": {
                    "line-color": ["match", ["get", "highway"],
                        "motorway", "#ffb366",
                        ["trunk", "motorway_link"], "#f7e06e",
                        ["primary", "secondary", "aeroway", "trunk_link"], "#fffbf8",
                        ["primary_link", "secondary_link", "tertiary", "tertiary_link"], "#ffffff",
                        "residential", "#ffffff",
                        "#ffffff"],
                    "line-width": [
                        "let",
                        "base", ["match", ["get", "highway"],
                            "motorway", 3.5,
                            ["trunk", "motorway_link"], 3,
                            ["primary", "secondary", "aeroway", "trunk_link"], 2.5,
                            ["primary_link", "secondary_link", "tertiary", "tertiary_link"], 1.75,
                            "residential", 1.5,
                            0.75],
                        ["interpolate", ["linear"], ["zoom"],
                            5, ["*", ["var", "base"], 0.5],
                            9, ["*", ["var", "base"], 1],
                            12, ["*", ["var", "base"], 2],
                            16, ["*", ["var", "base"], 2.5],
                            20, ["*", ["var", "base"], 3]
                        ]
                    ]
                }
            },
            {
                "id": "rail_old",
                "type": "line",
                "source": "osm",
                "source-layer": "rail",
                "filter": ["==", "rail", "old"],
                "paint": {
                    "line-color": rail,
                }
            },
            {
                "id": "rail_detail",
                "type": "line",
                "source": "osm",
                "source-layer": "rail",
                "filter": ["==", "rail", "detail"],
                "paint": {
                    "line-color": rail,
                }
            },
            {
                "id": "rail_secondary",
                "type": "line",
                "source": "osm",
                "source-layer": "rail",
                "filter": [
                    "all",
                    ["==", "rail", "secondary"],
                    ["any",
                        ["!has", "level"],
                        ["==", "level", level]
                    ]
                ],
                "paint": {
                    "line-color": rail,
                    "line-width": 1.15
                }
            },
            {
                "id": "rail_primary",
                "type": "line",
                "source": "osm",
                "source-layer": "rail",
                "filter": [
                    "all",
                    ["==", "rail", "primary"],
                    ["any",
                        ["!has", "level"],
                        ["==", "level", level]
                    ]
                ],
                "paint": {
                    "line-color": rail,
                    "line-width": 1.3
                }
            },
            {
                "id": "road-ref-shield",
                "type": "symbol",
                "source": "osm",
                "source-layer": "road",
                "minzoom": 6,
                "filter": ["all",
                    ["has", "ref"],
                    ["any", ["==", ["get", "highway"], "motorway"],
                        ["==", ["get", "highway"], "trunk"],
                        ["==", ["get", "highway"], "secondary"],
                        [">", ["zoom"], 11]]],
                "layout": {
                    "symbol-placement": "line",
                    "text-field": ["get", "ref"],
                    "text-font": ["Noto Sans Display Regular"],
                    "text-size": ["case", ["==", ["get", "highway"], "motorway"], 11, 10],
                    "text-justify": "center",
                    "text-rotation-alignment": "viewport",
                    "text-pitch-alignment": "viewport",
                    "icon-image": "shield",
                    "icon-text-fit": "both",
                    "icon-text-fit-padding": [.5, 4, .5, 4],
                    "icon-rotation-alignment": "viewport",
                    "icon-pitch-alignment": "viewport",
                },
                "paint": {
                    "text-color": "#333333",
                },
            },
            {
                "id": "road-name-text",
                "type": "symbol",
                "source": "osm",
                "source-layer": "road",
                "minzoom": 14,
                "layout": {
                    "symbol-placement": "line",
                    "text-field": ["get", "name"],
                    "text-font": ["Noto Sans Display Regular"],
                    "text-size": 9,
                },
                "paint": {
                    "text-halo-width": 11,
                    "text-halo-color": "white",
                    "text-color": "#333333"
                }
            },
            {

                "id": "towns",
                "type": "symbol",
                "source": "osm",
                "source-layer": "cities",
                "filter": ["!=", ["get", "place"], "city"],
                "layout": {
                    // "symbol-sort-key": ["get", "population"],
                    "text-field": ["get", "name"],
                    "text-font": ["Noto Sans Display Regular"],
                    "text-size": 12
                },
                "paint": {
                    "text-halo-width": 1,
                    "text-halo-color": "white",
                    "text-color": "#333333"
                }
            },
            {
                "id": "cities",
                "type": "symbol",
                "source": "osm",
                "source-layer": "cities",
                "filter": ["==", ["get", "place"], "city"],
                "layout": {
                    // "symbol-sort-key": ["get", "population"],
                    "text-field": ["get", "name"],
                    "text-font": ["Noto Sans Display Bold"],
                    "text-size": 18
                },
                "paint": {
                    "text-halo-width": 2,
                    "text-halo-color": "white",
                    "text-color": "#111111"
                }
            }
        ]
    };
}
