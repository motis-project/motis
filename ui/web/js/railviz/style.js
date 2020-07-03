var RailViz = RailViz || {};

/**
 * Shared (mapbox) style fragments / definitions
 */
RailViz.Style = (function () {
  function lineColor() {
    // prettier-ignore
    return ["case",
      ["coalesce", ["get", "stub"], true], "#DDDDDD",
      ["<", ["get", "min_class"], 6],      "#666666",
      ["<", ["get", "min_class"], 8],      "#999999",
      /* default */                        "#AAAAAA",
    ];
  }

  function lineWidth() {
    // prettier-ignore
    return  ["let",
      "w", ["case",
              ["coalesce", ["get", "stub"], true], 2,
              ["<", ["get", "min_class"], 6],      3.5,
              ["<", ["get", "min_class"], 8],      3.0,
              /* default */                        2.0,
            ],
      ["interpolate", ["linear"], ["zoom"],
        6,  ["*", ["var", "w"], .5],
        8,  ["*", ["var", "w"], .8],
        10, ["*", ["var", "w"], 1.]
      ],
    ];
  }

  function stationCircleRadius() {
    // prettier-ignore
    return  ["let",
      "w", ["case",
            ["<", ["get", "min_class"], 8],      1,
            ["<", ["get", "min_class"], 9],      0.9,
            ["<", ["get", "min_class"], 12],     0.7,
            /* default */                        0.5,
          ],
      ["interpolate", ["linear"], ["zoom"],
        6,  ["*", ["var", "w"], .75],
        8,  ["*", ["var", "w"], 1.25],
        10, ["*", ["var", "w"], 2.25],
        20, ["*", ["var", "w"], 4],
      ]
    ];
  }

  function stationCircleWidth() {
    // prettier-ignore
    return  ["let",
      "w", ["case",
            ["<", ["get", "min_class"], 8],      1.3,
            ["<", ["get", "min_class"], 9],      1.0,
            ["<", ["get", "min_class"], 12],     0.7,
            /* default */                        0.4,
          ],
      ["interpolate", ["linear"], ["zoom"],
        6,  ["*", ["var", "w"], 0.66],
        8,  ["*", ["var", "w"], 0.9],
        10, ["*", ["var", "w"], 1],
        20, ["*", ["var", "w"], 2],
      ]
    ];
  }

  function interpolateTrainScale(z) {
    const interpolate = (lb, ub, lv, uv) =>
      lv + ((z - lb) / (ub - lb)) * (uv - lv);

    if (z < 6) {
      return 0.5;
    } else if (z < 8) {
      return interpolate(6, 8, 0.5, 0.8);
    } else if (z < 10) {
      return interpolate(8, 10, 0.8, 1);
    } else {
      return 1;
    }
  }

  return {
    lineColor,
    lineWidth,
    stationCircleRadius,
    interpolateTrainScale,
    stationCircleWidth,
  };
})();
