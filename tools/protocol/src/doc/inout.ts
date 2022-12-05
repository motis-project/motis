import { SchemaType, SchemaTypes } from "../schema/types";
import path from "path";
import { DocField, DocPath, DocType, Documentation } from "./types";
import { DocContext, DocSchemaFile } from "./context";
import fs from "fs";
import {
  Document,
  isCollection,
  isScalar,
  isSeq,
  parse,
  parseDocument,
  Scalar,
  YAMLMap,
} from "yaml";
import { getOrCreateMap, removeUnknownKeys } from "../util/yaml";

export function readAndUpdateDoc(
  schema: SchemaTypes,
  baseDir: string,
  config: any
): Documentation {
  if (typeof config.schemas !== "string") {
    throw new Error("missing doc.schemas property in config");
  }
  if (typeof config.paths !== "string") {
    throw new Error("missing doc.paths property in config");
  }

  const schemasDir = path.resolve(baseDir, config.schemas);
  const pathsFile = path.resolve(baseDir, config.paths);

  const doc: Documentation = { types: new Map(), paths: [] };
  const ctx: DocContext = {
    schema,
    doc,
    schemasDir,
    pathsFile,
    schemaFiles: new Map(),
  };

  readAndUpdateSchemas(ctx);
  readPaths(ctx);

  return doc;
}

function readAndUpdateSchemas(ctx: DocContext) {
  for (const [fqtn, type] of ctx.schema.types) {
    const ns = type.ns.join(".");
    let file = ctx.schemaFiles.get(ns);
    if (file === undefined) {
      file = {
        path: path.resolve(ctx.schemasDir, ...type.ns) + ".yaml",
        namespace: ns,
        types: [],
      };
      ctx.schemaFiles.set(ns, file);
    }
    file.types.push(type.name);
  }

  for (const [ns, file] of ctx.schemaFiles) {
    readAndUpdateSchemaFile(ctx, file);
  }
}

function readAndUpdateSchemaFile(ctx: DocContext, file: DocSchemaFile) {
  console.log(file.path);
  let yd = new Document();
  if (fs.existsSync(file.path)) {
    yd = parseDocument(fs.readFileSync(file.path, { encoding: "utf8" }));
  }
  const root = getOrCreateMap(yd, yd, []);
  const hasExistingTypes = root.items.length > 0;
  const removedTypes = removeUnknownKeys(root, (key) =>
    ctx.schema.types.has(getFqtn(file, key))
  );
  if (removedTypes.length > 0) {
    console.log("removed unknown types:");
    for (const s of removedTypes) {
      console.log(`  ${getFqtn(file, s)}`);
    }
  }

  for (const tn of file.types) {
    const fqtn = getFqtn(file, tn);
    if (hasExistingTypes && !root.has(tn)) {
      console.log(`adding new type: ${fqtn}`);
    }
    const yt = getOrCreateMap(yd, root, [tn]);
    const schemaType = ctx.schema.types.get(fqtn);
    if (schemaType === undefined) {
      throw new Error(`unknown type: ${schemaType}`);
    }
    readAndUpdateType(ctx, file, yd, yt, schemaType);
  }

  fs.mkdirSync(path.dirname(file.path), { recursive: true });
  const out = fs.createWriteStream(file.path);
  out.write(yd.toString());
  out.end();
}

function readAndUpdateType(
  ctx: DocContext,
  file: DocSchemaFile,
  yd: Document,
  yt: YAMLMap,
  schemaType: SchemaType
) {
  const getOptStr = (map: YAMLMap, key: string) => {
    const val = map.get(key);
    if (val == undefined) {
      map.set(key, "TODO");
    }
    return typeof val === "string" && val !== "" && val !== "TODO"
      ? val
      : undefined;
  };

  const docType: DocType = {
    fqtn: [...schemaType.ns, ...schemaType.name].join("."),
    title: getOptStr(yt, "title"),
    description: getOptStr(yt, "description"),
    examples: [],
    tags: [],
  };

  const examples = yt.get("examples");
  if (isCollection(examples)) {
    docType.examples = examples.items;
  }

  const tags = yt.get("tags");
  if (isSeq(tags)) {
    docType.tags = tags.items
      .filter((n) => isScalar(n) && typeof n.value === "string")
      .map((n) => (n as Scalar<string>).value);
  }

  if (schemaType.type === "table") {
    docType.fields = new Map();
    const fields = getOrCreateMap(yd, yt, ["fields"]);
    removeUnknownKeys(fields, (key) =>
      schemaType.fields.some((f) => f.name === key)
    );
    for (const field of schemaType.fields) {
      const yf = getOrCreateMap(yd, fields, [field.name]);
      const docField: DocField = {
        name: field.name,
        description: getOptStr(yf, "description"),
      };
      const fieldExamples = yf.get("examples");
      if (isCollection(fieldExamples)) {
        docField.examples = fieldExamples.items;
      }
      docType.fields.set(field.name, docField);
    }
  }

  ctx.doc.types.set(docType.fqtn, docType);
}

function getFqtn(file: DocSchemaFile, key: string) {
  return file.namespace !== "" ? `${file.namespace}.${key}` : key;
}

function readPaths(ctx: DocContext) {
  if (!fs.existsSync(ctx.pathsFile)) {
    console.log(`warning: paths file does not exist: ${ctx.pathsFile}`);
    return;
  }
  const yd = parse(fs.readFileSync(ctx.pathsFile, { encoding: "utf8" }));
  if (typeof yd !== "object") {
    throw new Error("invalid paths file");
  }
  for (const path in yd) {
    const props = yd[path];
    if (typeof props !== "object") {
      throw new Error(`invalid paths file: path ${path}`);
    }
    const dp: DocPath = {
      path,
      summary: props.summary,
      description: props.description,
      input: props.input,
      output: [],
    };
    if (Array.isArray(props.output)) {
      dp.output = props.output.map((o: any) => {
        if (
          typeof o !== "object" ||
          typeof o.type !== "string" ||
          typeof o.description !== "string"
        ) {
          throw new Error(`invalid paths file: path ${path} (output)`);
        }
        return { type: o.type, description: o.description };
      });
    }
    ctx.doc.paths.push(dp);
  }
}
