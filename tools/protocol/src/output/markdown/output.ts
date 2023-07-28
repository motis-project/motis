import fs from "fs";
import path from "path";

import { Documentation } from "@/doc/types";
import { TypeFilter, includeType } from "@/filter/type-filter";
import { MarkdownContext, MarkdownFile } from "@/output/markdown/context";
import { getFilename, getTypeLink, toSingleLine } from "@/output/markdown/util";
import { FieldType, SchemaTypes } from "@/schema/types";
import { isRequired } from "@/util/required";

export function writeMarkdownOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  doc: Documentation,
  baseDir: string,
  config: object,
) {
  if (!("dir" in config) || typeof config.dir !== "string") {
    throw new Error("missing dir property in config");
  }

  const outputDir = path.resolve(baseDir, config.dir);
  console.log(`writing markdown output to: ${outputDir}`);

  const ctx: MarkdownContext = {
    schema,
    typeFilter,
    doc,
    files: new Map(),
    types: new Set(),
    outputDir,
  };

  for (const [fqtn, type] of schema.types) {
    if (!includeType(typeFilter, fqtn)) {
      continue;
    }
    const ns = type.ns.join(".");
    let file = ctx.files.get(ns);
    if (file === undefined) {
      file = {
        path: getFilename(ctx, type.ns),
        namespace: ns,
        types: [],
      };
      ctx.files.set(ns, file);
    }
    file.types.push(fqtn);
    ctx.types.add(fqtn);
  }

  for (const file of ctx.files.values()) {
    writeSchemaFile(ctx, file);
  }

  writePathsFile(ctx);
}

function writeSchemaFile(ctx: MarkdownContext, file: MarkdownFile) {
  fs.mkdirSync(path.dirname(file.path), { recursive: true });
  const out = fs.createWriteStream(file.path);

  for (const fqtn of file.types) {
    writeType(ctx, file, out, fqtn);
  }

  out.end();
}

function writeType(
  ctx: MarkdownContext,
  file: MarkdownFile,
  out: fs.WriteStream,
  fqtn: string,
) {
  const type = ctx.schema.types.get(fqtn);
  if (!type) {
    throw new Error(`undefined type ${fqtn}`);
  }

  const typeDoc = ctx.doc.types.get(fqtn);

  out.write(`## ${type.name}\n\n`);

  if (typeDoc?.description) {
    out.write(typeDoc.description.trim());
    out.write("\n\n");
  }

  switch (type.type) {
    case "enum":
      if (!typeDoc?.description) {
        out.write("Possible values:\n\n");
        for (const v of type.values) {
          out.write(`- \`${v.id}\`\n`);
        }
      }
      break;

    case "union":
      out.write("One of:\n\n");
      for (const v of type.values) {
        const fqtn = v.typeRef.resolvedFqtn;
        out.write(`- ${fqtn.join(".")}\n`);
      }
      break;

    case "table":
      for (const f of type.fields) {
        const fieldDoc = typeDoc?.fields?.get(f.name);
        const fieldTypeMd = fieldTypeToMarkdown(ctx, file, f.type);
        out.write(`- \`${f.name}\` (${fieldTypeMd})`);
        if (!isRequired(f.metadata)) {
          out.write(" (optional)");
        }
        f.type;
        if (fieldDoc?.description) {
          out.write(": ");
          out.write(toSingleLine(fieldDoc.description));
        }
        out.write("\n");
      }
      break;
  }

  out.write("\n\n");
}

function fieldTypeToMarkdown(
  ctx: MarkdownContext,
  file: MarkdownFile,
  fieldType: FieldType,
): string {
  switch (fieldType.c) {
    case "basic":
      switch (fieldType.type.sc) {
        case "bool":
          return "bool";
        case "int":
          return `${fieldType.type.unsigned ? "u" : ""}int${
            fieldType.type.bits
          }`;
        case "float":
          return fieldType.type.bits === 32 ? "float" : "double";
        case "string":
          return "string";
      }
      break;
    case "vector":
      return "array of " + fieldTypeToMarkdown(ctx, file, fieldType.type);
    case "custom": {
      return getTypeLink(ctx, fieldType.type.resolvedFqtn, file.namespace);
    }
  }
}

function writePathsFile(ctx: MarkdownContext) {
  if (ctx.doc.paths.length === 0) {
    return;
  }

  const fileName = path.resolve(ctx.outputDir, "paths.md");
  fs.mkdirSync(path.dirname(fileName), { recursive: true });
  const out = fs.createWriteStream(fileName);

  for (const path of ctx.doc.paths) {
    out.write(`## ${path.path}`);
    if (path.summary) {
      out.write(`: ${toSingleLine(path.summary)}`);
    }
    out.write("\n\n");

    if (path.description) {
      out.write(path.description.trim());
      out.write("\n\n");
    }

    // TODO: inline request + response types

    out.write("### Request\n\n");
    if (path.input) {
      const inputFqtn = path.input.split(".");
      const inputNamespace = inputFqtn.slice(0, -1).join(".");
      out.write(`Type: ${getTypeLink(ctx, inputFqtn, inputNamespace)}\n\n`);
    } else {
      out.write(
        `Type: ${getTypeLink(ctx, [
          "motis",
          "MotisNoMessage",
        ])} (or HTTP GET request)\n\n`,
      );
    }

    out.write("### Response\n\n");
    if (path.output) {
      const outputFqtn = path.output.type.split(".");
      const outputNamespace = outputFqtn.slice(0, -1).join(".");
      out.write(`Type: ${getTypeLink(ctx, outputFqtn, outputNamespace)}\n\n`);
      out.write(path.output.description.trim());
      out.write("\n\n");
    } else {
      out.write(`Type: ${getTypeLink(ctx, ["motis", "MotisNoMessage"])}\n\n`);
    }
  }

  out.end();
}
