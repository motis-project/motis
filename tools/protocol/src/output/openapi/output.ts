import { SchemaTypes } from "../../schema/types";
import { TypeFilter } from "../../filter/type-filter";
import * as path from "path";
import fs from "fs";
import { Document, isMap, isScalar, parseDocument, YAMLMap } from "yaml";
import { createJSContext, getJSONSchemaTypes } from "../json-schema/output";
import { JSONSchema } from "../json-schema/types";
import { OpenApiContext } from "./context";

export function writeOpenAPIOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  baseDir: string,
  config: any
) {
  if (typeof config.file !== "string") {
    throw new Error("missing file property in config");
  }
  if (typeof config["base-uri"] !== "string") {
    throw new Error("missing base-uri property in config");
  }
  if (typeof config.info !== "object") {
    throw new Error("missing info property in config");
  }

  const openApiFile = path.resolve(baseDir, config.file);

  const baseUri = new URL(config["base-uri"]);
  if (!baseUri.pathname.endsWith("/")) {
    baseUri.pathname += "/";
  }

  let doc = new Document();
  if (fs.existsSync(openApiFile)) {
    console.log(`loading existing open api specification: ${openApiFile}`);
    doc = parseDocument(fs.readFileSync(openApiFile, { encoding: "utf8" }));
  }

  const jsCtx = createJSContext(
    schema,
    typeFilter,
    baseUri,
    (fqtn) => `#/components/schemas/${fqtn.join(".")}`
  );
  const jsonSchema = getJSONSchemaTypes(jsCtx);

  const ctx: OpenApiContext = {
    schema,
    typeFilter,
    baseUri,
    openApiVersion: "3.1.0",
    jsonSchema,
    doc,
    includeIds: config["ids"] !== false,
  };

  if (doc.contents === null) {
    doc.contents = doc.createNode({});
  }
  if (!isMap(doc.contents)) {
    throw new Error("invalid open api yaml file: root is not a map");
  }

  if (doc.has("openapi")) {
    const existingVersion = doc.get("openapi");
    if (existingVersion !== ctx.openApiVersion) {
      throw new Error(`unsupported open api version: ${existingVersion}`);
    }
  } else {
    doc.set("openapi", ctx.openApiVersion);
  }

  for (const key in config.info) {
    doc.setIn(["info", key], config.info[key]);
  }

  const oaSchemas = getOrCreateMap(doc, doc, ["components", "schemas"]);
  const hasExistingSchemas = oaSchemas.items.length > 0;

  const removedSchemas = removeUnknownKeys(
    oaSchemas,
    (key) => key in jsonSchema
  );
  if (removedSchemas.length > 0) {
    console.log("removed unknown schemas:");
    for (const s of removedSchemas) {
      console.log(`  ${s}`);
    }
  }

  for (const fqtn in jsonSchema) {
    let oaSchema = oaSchemas.get(fqtn);
    if (oaSchema == undefined) {
      if (hasExistingSchemas) {
        console.log(`adding new schema: ${fqtn}`);
      }
      oaSchema = doc.createNode({});
      oaSchemas.set(fqtn, oaSchema);
    }
    if (!isMap(oaSchema)) {
      throw new Error(
        `invalid open api yaml file: schema is not a map: ${fqtn}`
      );
    }
    updateSchema(ctx, oaSchema, jsonSchema[fqtn]);
  }

  console.log(`writing open api specification: ${openApiFile}`);
  fs.mkdirSync(path.dirname(openApiFile), { recursive: true });
  const out = fs.createWriteStream(openApiFile);
  out.write(doc.toString());
  out.end();
}

function updateSchema(
  ctx: OpenApiContext,
  oaSchema: YAMLMap,
  jsonSchema: JSONSchema
) {
  function setKey(key: keyof JSONSchema) {
    if (key in jsonSchema) {
      oaSchema.set(key, jsonSchema[key]);
    } else {
      oaSchema.delete(key);
    }
  }

  if (ctx.includeIds) {
    setKey("$id");
  } else {
    oaSchema.delete("$id");
  }

  setKey("$ref");
  setKey("type");
  setKey("required");
  setKey("enum");
  setKey("const");
  setKey("minimum");
  setKey("maximum");

  if (jsonSchema.properties) {
    const jsProps = jsonSchema.properties;
    const oaProps = getOrCreateMap(ctx.doc, oaSchema, ["properties"]);
    removeUnknownKeys(oaProps, (key) => key in jsProps);

    for (const key in jsProps) {
      const jsProp = jsProps[key];
      const oaProp = getOrCreateMap(ctx.doc, oaProps, [key]);
      updateSchema(ctx, oaProp, jsProp);
    }
  } else {
    oaSchema.delete("properties");
  }

  if (jsonSchema.items) {
    updateSchema(
      ctx,
      getOrCreateMap(ctx.doc, oaSchema, ["items"]),
      jsonSchema.items
    );
  } else {
    oaSchema.delete("items");
  }

  // for now
  setKey("allOf");
  setKey("anyOf");
  setKey("oneOf");
  setKey("not");
  setKey("if");
  setKey("then");
  setKey("else");
}

function removeUnknownKeys(
  oaMap: YAMLMap,
  keep: (key: string) => boolean
): string[] {
  const unknownKeys: string[] = [];
  for (const pair of oaMap.items) {
    if (!isScalar(pair.key) || typeof pair.key.value !== "string") {
      throw new Error(
        `invalid open api yaml file: unsupported map key: ${pair.toJSON()}`
      );
    }
    if (!keep(pair.key.value)) {
      unknownKeys.push(pair.key.value);
    }
  }
  for (const s of unknownKeys) {
    oaMap.delete(s);
  }
  return unknownKeys;
}

function getOrCreateMap(
  doc: Document,
  parent: YAMLMap | Document,
  path: string[]
): YAMLMap {
  let map = parent.getIn(path);
  if (map == undefined) {
    map = doc.createNode({});
    parent.setIn(path, map);
  }
  if (!isMap(map)) {
    throw new Error(
      `invalid open api yaml file: ${path.join("/")} is not a map`
    );
  }
  return map;
}
