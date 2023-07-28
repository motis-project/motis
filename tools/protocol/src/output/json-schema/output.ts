import fs from "fs";
import path from "path";

import { TypeFilter, includeType } from "@/filter/type-filter";
import { JSContext } from "@/output/json-schema/context";
import { basicTypeToJS } from "@/output/json-schema/primitive-types";
import { JSONDiscriminator, JSONSchema } from "@/output/json-schema/types";
import {
  FieldType,
  SchemaType,
  SchemaTypes,
  TableType,
  TypeBase,
} from "@/schema/types";
import { isRequired } from "@/util/required";

const JSON_SCHEMA_URL = "https://json-schema.org/draft/2020-12/schema";

export function writeJsonSchemaOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  baseDir: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any,
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
    config["strict-unions"] !== false,
    config["types-in-unions"] !== false,
    false,
    false,
    !!config["explicit-additional-properties"],
    config["tagged-type-suffix"] || "T",
  );

  const { types: defs } = getJSONSchemaTypes(ctx);

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
          2,
        ),
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
  strictUnions = true,
  typesInUnions = true,
  includeOpenApiDiscriminators = false,
  constAsEnum = false,
  explicitAdditionalProperties = false,
  taggedTypeFnOrSuffix: ((fqtn: string[]) => string[]) | string = "T",
  typeKey = "_type",
): JSContext {
  const ctx: JSContext = {
    schema,
    typeFilter,
    baseUri,
    jsonSchema: new Map(),
    taggedToUntaggedType: new Map(),
    untaggedToTaggedType: new Map(),
    strictIntTypes,
    numberFormats,
    strictUnions,
    typesInUnions,
    getRefUrl: getRefUrl || ((fqtn) => getDefaultRefUrl(ctx, fqtn)),
    getTaggedType:
      typeof taggedTypeFnOrSuffix === "function"
        ? taggedTypeFnOrSuffix
        : (fqtn) => getDefaultTaggedType(fqtn, taggedTypeFnOrSuffix),
    typeKey,
    includeOpenApiDiscriminators,
    constAsEnum,
    explicitAdditionalProperties,
  };
  return ctx;
}

export interface JSONSchemaTypes {
  types: Record<string, JSONSchema>;
  taggedToUntaggedType: Map<string, string>; // tagged -> untagged fqtn
  untaggedToTaggedType: Map<string, string>; // untagged -> tagged fqtn
}

export function getJSONSchemaTypes(ctx: JSContext): JSONSchemaTypes {
  for (const [fqtn, type] of ctx.schema.types) {
    if (!includeType(ctx.typeFilter, fqtn)) {
      continue;
    }
    convertSchemaType(ctx, fqtn, type);
  }
  return {
    types: bundleDefs(ctx),
    taggedToUntaggedType: ctx.taggedToUntaggedType,
    untaggedToTaggedType: ctx.untaggedToTaggedType,
  };
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
      if (ctx.typesInUnions) {
        union.oneOf = [];
        const discriminator: JSONDiscriminator = {
          propertyName: ctx.typeKey,
          mapping: {},
        };
        for (const value of type.values) {
          const fqtn = value.typeRef.resolvedFqtn;
          const fqtnStr = fqtn.join(".");
          if (includeType(ctx.typeFilter, fqtnStr)) {
            const taggedFqtn = ctx.getTaggedType(fqtn);
            const refUrl = ctx.getRefUrl(taggedFqtn);
            union.oneOf.push({ $ref: refUrl });
            discriminator.mapping[fqtn[fqtn.length - 1]] = refUrl;
          }
        }
        if (ctx.includeOpenApiDiscriminators) {
          union.discriminator = discriminator;
        }
        ctx.jsonSchema.set(fqtn, union);
      } else {
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
      }
      break;
    }
    case "table": {
      const untagged = !ctx.typesInUnions || type.usedInTable;
      const tagged = ctx.typesInUnions && type.usedInUnion;
      if (untagged) {
        ctx.jsonSchema.set(
          fqtn,
          addTableProperties(
            ctx,
            type,
            {
              ...base,
              type: "object",
            },
            false,
          ),
        );
      }
      if (tagged) {
        const taggedBase = getTaggedBaseJSProps(ctx, type);
        const taggedType = ctx.getTaggedType(fqtn.split("."));
        const taggedTypeStr = taggedType.join(".");
        ctx.jsonSchema.set(
          taggedTypeStr,
          addTableProperties(
            ctx,
            type,
            {
              ...taggedBase,
              type: "object",
            },
            true,
          ),
        );
        ctx.taggedToUntaggedType.set(taggedTypeStr, fqtn);
        ctx.untaggedToTaggedType.set(fqtn, taggedTypeStr);
      }
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

function addTableProperties(
  ctx: JSContext,
  type: TableType,
  js: JSONSchema,
  tagged: boolean,
) {
  const props: Record<string, JSONSchema> = {};
  const unionCases: JSONSchema[] = [];
  const required: string[] = [];

  if (tagged) {
    props[ctx.typeKey] = getConstString(ctx, type.name);
    required.push(ctx.typeKey);
  }

  for (const field of type.fields) {
    const requiredField = isRequired(field.metadata);
    if (field.type.c === "custom") {
      const fqtn = field.type.type.resolvedFqtn.join(".");
      const resolvedType = ctx.schema.types.get(fqtn);
      if (!resolvedType) {
        throw new Error(
          `unknown type ${fqtn} (${[...type.ns, type.name].join(".")}#${
            field.name
          })`,
        );
      }
      if (resolvedType.type === "union" && !ctx.typesInUnions) {
        const tagName = `${field.name}_type`;
        if (requiredField) {
          required.push(tagName);
        }
        if (ctx.strictUnions) {
          if (requiredField) {
            required.push(field.name);
          }
          for (const value of resolvedType.values) {
            const fqtn = value.typeRef.resolvedFqtn;
            const fqtnStr = fqtn.join(".");
            if (includeType(ctx.typeFilter, fqtnStr)) {
              const fqtn = value.typeRef.resolvedFqtn;
              const tag = fqtn[fqtn.length - 1];
              unionCases.push({
                if: {
                  properties: { [tagName]: getConstString(ctx, tag) },
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
    if (requiredField) {
      required.push(field.name);
    }
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
  if (ctx.explicitAdditionalProperties) {
    js.additionalProperties = true;
  }
  return js;
}

function getBaseJSProps(ctx: JSContext, type: TypeBase): JSONSchema {
  return { $id: ctx.baseUri.href + [...type.ns, type.name].join("/") };
}

function getTaggedBaseJSProps(ctx: JSContext, type: TypeBase): JSONSchema {
  return {
    $id:
      ctx.baseUri.href + ctx.getTaggedType([...type.ns, type.name]).join("/"),
  };
}

function getDefaultRefUrl(ctx: JSContext, fqtn: string[], absolute = false) {
  return (absolute ? ctx.baseUri.href : ctx.baseUri.pathname) + fqtn.join("/");
}

function getUnionTagRefUrl(ctx: JSContext, baseFqtn: string[]) {
  const fqtn = [...baseFqtn];
  fqtn[fqtn.length - 1] += "Type";
  return ctx.getRefUrl(fqtn);
}

function getDefaultTaggedType(baseFqtn: string[], taggedTypeSuffix: string) {
  const fqtn = [...baseFqtn];
  fqtn[fqtn.length - 1] += taggedTypeSuffix;
  return fqtn;
}

function getConstString(ctx: JSContext, value: string): JSONSchema {
  if (ctx.constAsEnum) {
    return { type: "string", enum: [value] };
  } else {
    return { type: "string", const: value };
  }
}

function bundleDefs(ctx: JSContext) {
  const defs: Record<string, JSONSchema> = {};
  for (const [fqtn, schema] of ctx.jsonSchema) {
    defs[fqtn] = schema;
  }
  return defs;
}
