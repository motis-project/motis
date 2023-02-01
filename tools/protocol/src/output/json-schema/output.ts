import fs from "fs";
import path from "path";

import { TypeFilter, includeType } from "@/filter/type-filter";
import { JSContext } from "@/output/json-schema/context";
import { basicTypeToJS } from "@/output/json-schema/primitive-types";
import { JSONSchema } from "@/output/json-schema/types";
import {
  FieldType,
  SchemaType,
  SchemaTypes,
  TableType,
  TypeBase,
} from "@/schema/types";

const JSON_SCHEMA_URL = "https://json-schema.org/draft/2020-12/schema";

export function writeJsonSchemaOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  baseDir: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any
) {
  if (typeof config["base-uri"] !== "string") {
    throw new Error("missing base-uri property in config");
  }
  if (typeof config.dir !== "string" && typeof config.bundle !== "string") {
    throw new Error("missing dir or bundle property in config");
  }

  const outputDir = config.dir ? path.resolve(baseDir, config.dir) : null;
  const bundleFile = config.bundle
    ? path.resolve(baseDir, config.bundle)
    : null;

  const baseUri = new URL(config["base-uri"]);
  if (!baseUri.pathname.endsWith("/")) {
    baseUri.pathname += "/";
  }

  const ctx = createJSContext(
    schema,
    typeFilter,
    baseUri,
    null,
    !!config["strict-int-types"],
    !!config["number-formats"],
    config["strict-unions"] !== false
  );

  const defs = getJSONSchemaTypes(ctx);

  if (outputDir) {
    console.log(`writing json schema files to: ${outputDir}`);
    for (const fqtn in defs) {
      const fileName =
        path.resolve(outputDir, ...fqtn.split(".")) + ".schema.json";
      fs.mkdirSync(path.dirname(fileName), { recursive: true });
      const out = fs.createWriteStream(fileName);
      out.write(
        JSON.stringify(
          {
            $schema: JSON_SCHEMA_URL,
            ...defs[fqtn],
          },
          null,
          2
        )
      );
      out.end();
    }
  }
  if (bundleFile) {
    console.log(`writing json schema bundle to: ${bundleFile}`);
    fs.mkdirSync(path.dirname(bundleFile), { recursive: true });
    const out = fs.createWriteStream(bundleFile);
    const js: JSONSchema = {
      $schema: JSON_SCHEMA_URL,
      $defs: defs,
    };
    if (ctx.schema.rootType) {
      js.$ref = getDefaultRefUrl(ctx, ctx.schema.rootType.resolvedFqtn, true);
    }
    out.write(JSON.stringify(js, null, 2));
    out.end();
  }
}

export function createJSContext(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  baseUri: URL,
  getRefUrl: ((fqtn: string[]) => string) | null = null,
  strictIntTypes = false,
  numberFormats = false,
  strictUnions = true
): JSContext {
  const ctx: JSContext = {
    schema,
    typeFilter,
    baseUri,
    jsonSchema: new Map(),
    strictIntTypes,
    numberFormats,
    strictUnions,
    getRefUrl: getRefUrl || ((fqtn) => getDefaultRefUrl(ctx, fqtn)),
  };
  return ctx;
}

export function getJSONSchemaTypes(ctx: JSContext): Record<string, JSONSchema> {
  for (const [fqtn, type] of ctx.schema.types) {
    if (!includeType(ctx.typeFilter, fqtn)) {
      continue;
    }
    convertSchemaType(ctx, fqtn, type);
  }
  return bundleDefs(ctx);
}

function convertSchemaType(ctx: JSContext, fqtn: string, type: SchemaType) {
  const base = getBaseJSProps(ctx, type);
  switch (type.type) {
    case "enum":
      ctx.jsonSchema.set(fqtn, {
        ...base,
        type: "string",
        enum: type.values.map((v) => v.id),
      });
      break;
    case "union": {
      const union: JSONSchema = { ...base };
      const unionTags: JSONSchema = {
        $id: ctx.baseUri.href + [...type.ns, `${type.name}Type`].join("/"),
        type: "string",
      };
      union.anyOf = [];
      unionTags.enum = [];
      for (const value of type.values) {
        const fqtn = value.typeRef.resolvedFqtn;
        const fqtnStr = fqtn.join(".");
        if (includeType(ctx.typeFilter, fqtnStr)) {
          const fqtn = value.typeRef.resolvedFqtn;
          union.anyOf.push({ $ref: ctx.getRefUrl(fqtn) });
          unionTags.enum.push(fqtn[fqtn.length - 1]);
        }
      }
      ctx.jsonSchema.set(fqtn, union);
      ctx.jsonSchema.set(`${fqtn}Type`, unionTags);
      break;
    }
    case "table": {
      ctx.jsonSchema.set(
        fqtn,
        addTableProperties(ctx, type, {
          ...base,
          type: "object",
        })
      );
      break;
    }
  }
}

function fieldTypeToJS(ctx: JSContext, type: FieldType): JSONSchema {
  switch (type.c) {
    case "basic":
      return basicTypeToJS(ctx, type.type);
    case "vector":
      return { type: "array", items: fieldTypeToJS(ctx, type.type) };
    case "custom":
      return { $ref: ctx.getRefUrl(type.type.resolvedFqtn) };
  }
  throw new Error(`unhandled field type: ${JSON.stringify(type)}`);
}

function addTableProperties(ctx: JSContext, type: TableType, js: JSONSchema) {
  const props: { [name: string]: JSONSchema } = {};
  const unionCases: JSONSchema[] = [];
  const required: string[] = [];
  for (const field of type.fields) {
    if (field.type.c === "custom") {
      const fqtn = field.type.type.resolvedFqtn.join(".");
      const resolvedType = ctx.schema.types.get(fqtn);
      if (!resolvedType) {
        throw new Error(
          `unknown type ${fqtn} (${[...type.ns, type.name].join(".")}#${
            field.name
          })`
        );
      }
      if (resolvedType.type === "union") {
        const tagName = `${field.name}_type`;
        required.push(tagName);
        if (ctx.strictUnions) {
          required.push(field.name);
          for (const value of resolvedType.values) {
            const fqtn = value.typeRef.resolvedFqtn;
            const fqtnStr = fqtn.join(".");
            if (includeType(ctx.typeFilter, fqtnStr)) {
              const fqtn = value.typeRef.resolvedFqtn;
              const tag = fqtn[fqtn.length - 1];
              unionCases.push({
                if: {
                  properties: { [tagName]: { type: "string", const: tag } },
                },
                then: {
                  properties: {
                    [field.name]: { $ref: ctx.getRefUrl(fqtn) },
                  },
                },
              });
            }
          }
          continue;
        } else {
          props[tagName] = {
            $ref: getUnionTagRefUrl(ctx, field.type.type.resolvedFqtn),
          };
        }
      }
    }
    props[field.name] = fieldTypeToJS(ctx, field.type);
    required.push(field.name);
  }
  if (Object.keys(props).length > 0) {
    js.properties = props;
  }
  if (unionCases.length > 0) {
    js.allOf = unionCases;
  }
  if (required.length > 0) {
    js.required = required;
  }
  return js;
}

function getBaseJSProps(ctx: JSContext, type: TypeBase): JSONSchema {
  return { $id: ctx.baseUri.href + [...type.ns, type.name].join("/") };
}

function getDefaultRefUrl(ctx: JSContext, fqtn: string[], absolute = false) {
  return (absolute ? ctx.baseUri.href : ctx.baseUri.pathname) + fqtn.join("/");
}

function getUnionTagRefUrl(ctx: JSContext, baseFqtn: string[]) {
  const fqtn = [...baseFqtn];
  fqtn[fqtn.length - 1] += "Type";
  return ctx.getRefUrl(fqtn);
}

function bundleDefs(ctx: JSContext) {
  const defs: Record<string, JSONSchema> = {};
  for (const [fqtn, schema] of ctx.jsonSchema) {
    defs[fqtn] = schema;
  }
  return defs;
}
