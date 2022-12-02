const colors = require("tailwindcss/colors");

module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
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
        "db-warm-gray": {
          100: "#F5F4F1",
          200: "#DDDED6",
          300: "#BCBBB2",
          400: "#9C9A8E",
          500: "#858379",
          600: "#747067",
          700: "#4F4B41",
          800: "#38342F",
        },
        "db-cool-gray": {
          100: "#F0F3F5",
          200: "#D7DCE1",
          300: "#AFB4BB",
          400: "#878C96",
          500: "#646973",
          600: "#3C414B",
          700: "#282D37",
          800: "#131821",
        },
        "db-gray": "#cecece",
      },
    },
  },
  plugins: [require("@tailwindcss/forms"), require("@headlessui/tailwindcss")],
};
