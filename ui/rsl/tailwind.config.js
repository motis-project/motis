const colors = require("tailwindcss/colors");

module.exports = {
  mode: "jit",
  purge: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      colors: {
        gray: colors.blueGray,
        class: {
          air: "#90a4ae",
          ice: "#9c27b0",
          ic: "#e91e63",
          coach: "#9ccc65",
          n: "#1a237e",
          re: "#f44336",
          rb: "#f44336",
          s: "#4caf50",
          u: "#3f51b5",
          str: "#ff9800",
          bus: "#ff9800",
          ship: "#00acc1",
          other: "#9e9e9e",
        },
        // https://marketingportal.extranet.deutschebahn.com/de/farbe
        "db-red": {
          100: "#FEE6E6",
          200: "#FCC8C3",
          300: "#FA9090",
          400: "#F75056",
          500: "#EC0016",
          600: "#C50014",
          700: "#9B000E",
          800: "#740009",
        },
        "db-gray": "#cecece"
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [require("@tailwindcss/forms")],
};
