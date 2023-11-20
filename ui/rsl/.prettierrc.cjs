module.exports = {
  plugins: [
    "@trivago/prettier-plugin-sort-imports",
    "prettier-plugin-tailwindcss",
  ],
  // https://github.com/trivago/prettier-plugin-sort-imports
  importOrder: [
    "^@/api/protocol/(.*)$",
    "^@/api/(.*)$",
    "^@/data/(.*)$",
    "^@/util/(.*)$",
    "^@/components/(.*)$",
    "^@/(.*)$",
    "^[./]",
  ],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
};
