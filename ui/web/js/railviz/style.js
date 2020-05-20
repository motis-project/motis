var RailViz = RailViz || {};

/**
 * Shared (mapbox) style fragments / definitions
 */
RailViz.Style = (function () {
  function lineColor() {
    // prettier-ignore
    return ["case",
      ["get", "stub"],                "hsl(0, 0, 70%)", 
      ["<", ["get", "min_class"], 6], "hsl(0, 0, 30%)",
      ["<", ["get", "min_class"], 8], "hsl(0, 0, 40%)",
      /* default */                   "hsl(0, 0, 50%)",
    ];
  }

  function lineWidth() {
    // prettier-ignore
    return  ["let",
      "w", ["case",
              ["get", "stub"],                2,
              ["<", ["get", "min_class"], 6], 4,
              ["<", ["get", "min_class"], 8], 3.25,
              /* default */                   2.5,
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
    return ["interpolate", ["linear"], ["zoom"],
      6,  .75,
      8,  1.25,
      10, 2.25,
      20, 4,
    ];
  }

  function interpolateTrainScale(z) {
    const interpolate = (lb, ub, lv, uv) =>
      lv + ((z - lb) / (ub - lb)) * (uv - lv);

    if (z < 6) {
      return 0.5;
    } else if (z < 8) {
      return interpolate(6, 8 , .5, .8);
    } else if (z < 10) {
      return interpolate(8, 10, .8, 1);
    } else {
      return 1;
    }
  }

  return { lineColor, lineWidth, stationCircleRadius, interpolateTrainScale };
})();
