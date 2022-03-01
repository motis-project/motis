module.exports = {
  plugins: [
    require.resolve("@trivago/prettier-plugin-sort-imports"),
    require.resolve("prettier-plugin-alias-imports"),
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
