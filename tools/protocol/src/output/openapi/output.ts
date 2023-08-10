import fs from "fs";
import * as path from "path";
import { Document, YAMLMap, isScalar } from "yaml";

import { DocField, DocType, Documentation } from "@/doc/types";
import { TypeFilter } from "@/filter/type-filter";
import {
  createJSContext,
  getJSONSchemaTypes,
} from "@/output/json-schema/output";
import { JSONSchema } from "@/output/json-schema/types";
import { OPEN_API_VERSIONS, OpenApiContext } from "@/output/openapi/context";
import { SchemaTypes } from "@/schema/types";
import { compareFqtns, sortTypes } from "@/util/sort";
import { createMap } from "@/util/yaml";

export function writeOpenAPIOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  doc: Documentation,
  baseDir: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any,
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
  if (typeof config.version !== "string") {
    throw new Error("missing version property in config");
  }

  const openApiFile = path.resolve(baseDir, config.file);
  const openApiVersion = config.version;
  if (!OPEN_API_VERSIONS.includes(openApiVersion)) {
    throw new Error(`unsupported open api version: ${openApiVersion}`);
  }

  const baseUri = new URL(config["base-uri"]);
  if (!baseUri.pathname.endsWith("/")) {
    baseUri.pathname += "/";
  }

  const typesInUnions = config["types-in-unions"] !== false;
  const msgContentOnly = typesInUnions && config["msg-content-only"] !== false;
  const explicitAdditionalProperties =
    config["explicit-additional-properties"] !== false;

  const jsCtx = createJSContext(
    schema,
    typeFilter,
    baseUri,
    getRefUrl,
    false,
    true,
    false,
    typesInUnions,
    typesInUnions,
    true,
    explicitAdditionalProperties,
  );
  const jsonSchema = getJSONSchemaTypes(jsCtx);

  const yd = new Document();

  const ctx: OpenApiContext = {
    schema,
    typeFilter,
    baseUri,
    openApiVersion,
    jsonSchema,
    doc,
    yd,
    includeIds: config.ids !== false,
    typesInUnions,
    msgContentOnly,
  };

  yd.contents = yd.createNode({});
  yd.set("openapi", ctx.openApiVersion);

  function copyBlock(key: string) {
    if (config[key]) {
      yd.set(key, config[key]);
    }
  }

  copyBlock("info");
  copyBlock("externalDocs");
  copyBlock("servers");

  writeTags(ctx);
  writePaths(ctx);

  const oaSchemas = createMap(yd, yd, ["components", "schemas"]);

  const types = Object.keys(jsonSchema.types);
  sortTypes(types);

  for (const fqtn of types) {
    const origFqtn = jsonSchema.taggedToUntaggedType.get(fqtn) || fqtn;
    const oaSchema = createMap(yd, oaSchemas, [fqtn]);
    const typeDoc = ctx.doc.types.get(origFqtn);
    writeSchema(
      ctx,
      oaSchema,
      jsonSchema.types[fqtn],
      typeDoc,
      undefined,
      fqtn,
    );
  }

  oaSchemas.items.sort((a, b) =>
    isScalar(a.key) &&
    typeof a.key.value === "string" &&
    isScalar(b.key) &&
    typeof b.key.value === "string"
      ? compareFqtns(a.key.value, b.key.value)
      : 0,
  );

  console.log(`writing open api specification: ${openApiFile}`);
  fs.mkdirSync(path.dirname(openApiFile), { recursive: true });
  const out = fs.createWriteStream(openApiFile);
  out.write(yd.toString({ lineWidth: 100 }));
  out.end();
}

function getRefUrl(fqtn: string[]) {
  return `#/components/schemas/${fqtn.join(".")}`;
}

function writeTags(ctx: OpenApiContext) {
  const tags = ctx.doc.tags;
  if (tags.length > 0) {
    const oaTags = ctx.yd.createNode(tags);
    ctx.yd.set("tags", oaTags);
  }
}

function writeResponse(
  ctx: OpenApiContext,
  oaResponses: YAMLMap,
  code: string,
  fqtn: string,
  description: string,
) {
  const resType = fqtn.split(".");
  const resTypeName = resType[resType.length - 1];
  const oaResponse = createMap(ctx.yd, oaResponses, [code]);
  oaResponse.set("description", description);
  const oaResponseSchema = createMap(ctx.yd, oaResponse, [
    "content",
    "application/json",
    "schema",
  ]);
  if (ctx.msgContentOnly) {
    const taggedType = ctx.jsonSchema.untaggedToTaggedType.get(fqtn);
    if (!taggedType) {
      throw new Error(`OpenAPI: No tagged type for ${fqtn} found`);
    }
    oaResponseSchema.set("$ref", getRefUrl(taggedType.split(".")));
  } else {
    oaResponseSchema.set("type", "object");
    oaResponseSchema.set("required", ["content_type", "content"]);
    oaResponseSchema.set("properties", {
      destination: {
        type: "object",
        required: ["target"],
        properties: {
          target: { type: "string", enum: [""] },
          type: { type: "string", enum: ["Module"] },
        },
      },
      content_type: {
        type: "string",
        enum: [resTypeName],
      },
      content: {
        $ref: getRefUrl(resType),
      },
      id: { type: "integer", format: "int32" },
    });
  }
}

function writePaths(ctx: OpenApiContext) {
  const oaPaths = createMap(ctx.yd, ctx.yd, ["paths"]);
  for (const path of ctx.doc.paths) {
    const oaPath = createMap(ctx.yd, oaPaths, [path.path]);
    const post = !!path.input;
    const oaOperation = createMap(ctx.yd, oaPath, [post ? "post" : "get"]);
    const operationId =
      path.operationId ??
      path.path
        .substring(1)
        .replaceAll(/[/_]+(.)/g, (_, p1) => p1.toUpperCase());
    oaOperation.set("operationId", operationId);
    if (path.summary) {
      oaOperation.set("summary", path.summary);
    }
    if (path.description) {
      oaOperation.set("description", path.description);
    }
    if (path.tags.length > 0) {
      oaOperation.set("tags", path.tags);
    }
    if (path.deprecated) {
      oaOperation.set("deprecated", true);
    }

    if (path.input) {
      const reqFqtn = path.input;
      const oaRequest = createMap(ctx.yd, oaOperation, ["requestBody"]);
      oaRequest.set("required", true);
      const oaRequestSchema = createMap(ctx.yd, oaRequest, [
        "content",
        "application/json",
        "schema",
      ]);
      if (ctx.msgContentOnly) {
        const taggedType = ctx.jsonSchema.untaggedToTaggedType.get(reqFqtn);
        if (!taggedType) {
          throw new Error(`OpenAPI: No tagged type for ${reqFqtn} found`);
        }
        oaRequestSchema.set("$ref", getRefUrl(taggedType.split(".")));
      } else {
        const reqType = reqFqtn.split(".");
        const reqTypeName = reqType[reqType.length - 1];
        oaRequestSchema.set("type", "object");
        oaRequestSchema.set("required", [
          "destination",
          "content_type",
          "content",
        ]);
        oaRequestSchema.set("properties", {
          destination: {
            type: "object",
            required: ["target"],
            properties: {
              target: { type: "string", enum: [path.path] },
              type: { type: "string", enum: ["Module"] },
            },
          },
          content_type: {
            type: "string",
            enum: [reqTypeName],
          },
          content: {
            $ref: getRefUrl(reqType),
          },
          id: { type: "integer", format: "int32" },
        });
      }
    }

    const oaResponses = createMap(ctx.yd, oaOperation, ["responses"]);

    writeResponse(
      ctx,
      oaResponses,
      "200",
      path.output?.type ?? "motis.MotisSuccess",
      path.output?.description ?? "Empty response",
    );

    writeResponse(ctx, oaResponses, "500", "motis.MotisError", "Error");
  }
}

function writeSchema(
  ctx: OpenApiContext,
  oaSchema: YAMLMap,
  jsonSchema: JSONSchema,
  typeDoc?: DocType | undefined,
  fieldDoc?: DocField | undefined,
  fqtn?: string | undefined,
) {
  function setKey(key: keyof JSONSchema) {
    if (key in jsonSchema) {
      oaSchema.set(key, jsonSchema[key]);
    }
  }

  setKey("$ref");

  if (ctx.openApiVersion === "3.0.3" && "$ref" in jsonSchema) {
    // no $ref siblings allowed in OpenAPI 3.0
    return;
  }

  if (ctx.includeIds) {
    setKey("$id");
  }
  if (fqtn) {
    oaSchema.set("title", fqtn);
  }

  setKey("type");

  if (typeDoc) {
    if (typeDoc.description) {
      oaSchema.set("description", typeDoc.description);
    }
    if (typeDoc.tags && typeDoc.tags.length > 0) {
      oaSchema.set("tags", typeDoc.tags);
    }
    if (typeDoc.examples && typeDoc.examples.length > 0) {
      if (ctx.openApiVersion === "3.0.3") {
        oaSchema.set("example", typeDoc.examples[0]);
      } else {
        oaSchema.set("examples", typeDoc.examples);
      }
    }
  }
  if (fieldDoc) {
    if (fieldDoc.description) {
      oaSchema.set("description", fieldDoc.description);
    }
    if (fieldDoc.examples && fieldDoc.examples.length > 0) {
      if (ctx.openApiVersion === "3.0.3") {
        oaSchema.set("example", fieldDoc.examples[0]);
      } else {
        oaSchema.set("examples", fieldDoc.examples);
      }
    }
  }

  setKey("required");
  setKey("enum");
  setKey("const");
  setKey("format");
  setKey("minimum");
  setKey("maximum");

  if (jsonSchema.properties) {
    const jsProps = jsonSchema.properties;
    const oaProps = createMap(ctx.yd, oaSchema, ["properties"]);

    for (const key in jsProps) {
      const jsProp = jsProps[key];
      const oaProp = createMap(ctx.yd, oaProps, [key]);
      let fieldDoc = typeDoc?.fields?.get(key);
      if (!fieldDoc && !ctx.typesInUnions && key.endsWith("_type")) {
        fieldDoc = {
          name: key,
          description: `Type of the \`${key.replace(/_type$/, "")}\` field`,
        };
      }
      writeSchema(ctx, oaProp, jsProp, undefined, fieldDoc);
    }
  }

  if (jsonSchema.items) {
    writeSchema(ctx, createMap(ctx.yd, oaSchema, ["items"]), jsonSchema.items);
  }

  // for now
  setKey("allOf");
  setKey("anyOf");
  setKey("oneOf");
  setKey("discriminator");
  setKey("not");
  setKey("if");
  setKey("then");
  setKey("else");
  setKey("additionalProperties");
}
