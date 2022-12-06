import path from "path";
import fs from "fs";
import { resolveSchemaTypes } from "./schema/resolver";
import { parse } from "yaml";
import { parseTypeFilter } from "./filter/parse-filter";
import { writeTypeScriptOutput } from "./output/typescript/output";
import { writeJsonSchemaOutput } from "./output/json-schema/output";
import { writeOpenAPIOutput } from "./output/openapi/output";
import { readAndUpdateDoc } from "./doc/inout";

let baseDir = process.cwd();
const argv = process.argv.slice(2);
if (argv.length > 1) {
  console.log(
    `usage: ${process.argv.slice(0, 2).join(" ")} [protocol.config.yaml]`
  );
  process.exit(1);
}
const configFile = path.resolve(
  baseDir,
  argv.length === 1 ? argv[0] : "protocol.config.yaml"
);
if (!fs.existsSync(configFile)) {
  console.log(`config file not found: ${configFile}`);
  process.exit(2);
}
const config = parse(fs.readFileSync(configFile, { encoding: "utf8" }));

if (
  typeof config.input !== "string" ||
  typeof config.output !== "object" ||
  typeof config.doc !== "object"
) {
  console.log(`invalid config file: ${configFile}`);
  process.exit(3);
}

baseDir = path.dirname(configFile);
const rootInputFile = path.resolve(baseDir, config.input);
const rootInputPath = path.dirname(rootInputFile);

console.log(`base dir: ${baseDir}`);
console.log(`root input dir:  ${rootInputPath}`);
console.log(`root input file: ${rootInputFile}`);

const schema = resolveSchemaTypes(rootInputPath, rootInputFile);

const doc = readAndUpdateDoc(schema, baseDir, config.doc);

for (const outputName in config.output) {
  const output = config.output[outputName];
  console.log(`\n[${outputName}]`);
  try {
    const typeFilter = parseTypeFilter(output);
    switch (output.format) {
      case "typescript":
        writeTypeScriptOutput(schema, typeFilter, baseDir, output);
        break;
      case "json-schema":
        writeJsonSchemaOutput(schema, typeFilter, baseDir, output);
        break;
      case "openapi":
        writeOpenAPIOutput(schema, typeFilter, doc, baseDir, output);
        break;
      default:
        console.log(`unknown output format ${output.format}`);
        break;
    }
  } catch (e) {
    console.error(e);
  }
}
