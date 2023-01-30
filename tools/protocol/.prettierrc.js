module.exports = {
  plugins: [require.resolve("@trivago/prettier-plugin-sort-imports")],
  // https://github.com/trivago/prettier-plugin-sort-imports
  importOrder: ["^@/(.*)$", "^[./]"],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
};
