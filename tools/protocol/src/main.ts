import { Command } from "commander";
import fs from "fs";
import path from "path";
import { parse } from "yaml";

import { readAndUpdateDoc } from "@/doc/inout";
import { parseTypeFilter } from "@/filter/parse-filter";
import { writeJsonSchemaOutput } from "@/output/json-schema/output";
import { writeMarkdownOutput } from "@/output/markdown/output";
import { writeOpenAPIOutput } from "@/output/openapi/output";
import { writeTypeScriptOutput } from "@/output/typescript/output";
import { resolveSchemaTypes } from "@/schema/resolver";

const program = new Command();
program
  .name("pnpm start")
  .argument("[config]", "path to config file", "protocol.config.yaml")
  .option("-o, --output <name...>", "outputs to generate")
  .option("-s, --skip <name...>", "outputs to skip")
  .action(main);

async function main(configName: string) {
  const opts = program.opts();
  const includeOutputs: string[] | undefined = opts.output;
  const skipOutputs: string[] | undefined = opts.skip;

  let baseDir = process.cwd();
  const configFile = path.resolve(baseDir, configName);
  if (!fs.existsSync(configFile)) {
    console.log(`config file not found: ${configFile}`);
    process.exit(2);
  }
  const config = parse(fs.readFileSync(configFile, { encoding: "utf8" }));

  if (
    !("input" in config) ||
    !("output" in config) ||
    !("doc" in config) ||
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
    if (
      (includeOutputs && !includeOutputs.includes(outputName)) ||
      (skipOutputs && skipOutputs.includes(outputName))
    ) {
      continue;
    }
    const output = config.output[outputName];
    console.log(`\n[${outputName}]`);
    try {
      const typeFilter = parseTypeFilter(output);
      switch (output.format) {
        case "typescript":
          await writeTypeScriptOutput(schema, typeFilter, baseDir, output);
          break;
        case "json-schema":
          writeJsonSchemaOutput(schema, typeFilter, baseDir, output);
          break;
        case "openapi":
          writeOpenAPIOutput(schema, typeFilter, doc, baseDir, output);
          break;
        case "markdown":
          writeMarkdownOutput(schema, typeFilter, doc, baseDir, output);
          break;
        default:
          console.log(`unknown output format ${output.format}`);
          break;
      }
    } catch (e) {
      console.error(e);
    }
  }
}

program.parse();
