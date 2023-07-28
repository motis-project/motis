import fs from "fs";
import * as path from "path";
import * as prettier from "prettier";

import { TypeFilter, includeType } from "@/filter/type-filter";
import { TSContext, TSFile } from "@/output/typescript/context";
import { getFilename } from "@/output/typescript/filenames";
import { TSInclude, collectIncludes } from "@/output/typescript/includes";
import { getUnionTagTypeName } from "@/output/typescript/util";
import { FieldType, SchemaTypes, TableType, UnionValue } from "@/schema/types";
import { isRequired } from "@/util/required";

export async function writeTypeScriptOutput(
  schema: SchemaTypes,
  typeFilter: TypeFilter,
  baseDir: string,
  config: object,
) {
  if (!("dir" in config) || typeof config.dir !== "string") {
    throw new Error("missing dir property in config");
  }
  const outputDir = path.resolve(baseDir, config.dir);
  console.log(`writing typescript output to: ${outputDir}`);

  const files = new Map<string, TSFile>();
  const ctx: TSContext = {
    schema,
    typeFilter,
    files,
    types: new Set(),
    header: "",
    outputDir,
    importBase: null,
    usePrettier: false,
    prettierOptions: {},
  };

  if ("header" in config && typeof config.header === "string") {
    ctx.header = `${config.header.trim()}\n\n`;
  }

  if ("import-base" in config && typeof config["import-base"] === "string") {
    ctx.importBase = config["import-base"];
  }

  if ("prettier" in config && config.prettier === true) {
    ctx.usePrettier = true;
    const resolvedOptions = await prettier.resolveConfig(ctx.outputDir);
    if (resolvedOptions != null) {
      ctx.prettierOptions = resolvedOptions;
    }
    ctx.prettierOptions.parser = "typescript";
  }

  for (const [fqtn, type] of schema.types) {
    if (!includeType(typeFilter, fqtn)) {
      continue;
    }
    const ns = type.ns.join(".");
    let file = files.get(ns);
    if (file === undefined) {
      file = {
        path: getFilename(ctx, type.ns),
        namespace: ns,
        types: [],
      };
      files.set(ns, file);
    }
    file.types.push(fqtn);
    ctx.types.add(fqtn);
  }

  for (const file of files.values()) {
    await writeFile(ctx, file);
  }
}

function getImportPath(ctx: TSContext, file: TSFile, include: TSInclude) {
  if (ctx.importBase) {
    return (
      ctx.importBase +
      path
        .relative(ctx.outputDir, include.filename)
        .replace("\\", "/")
        .replace(/\.ts$/, "")
    );
  } else {
    let p = path
      .relative(path.dirname(file.path), include.filename)
      .replace("\\", "/")
      .replace(/\.ts$/, "");
    if (!p.startsWith("../")) {
      p = "./" + p;
    }
    return p;
  }
}

async function writeFile(ctx: TSContext, file: TSFile) {
  console.log(`writing ${file.path}: ${file.types.length} types`);
  fs.mkdirSync(path.dirname(file.path), { recursive: true });
  let out = ctx.header;

  const includes = collectIncludes(ctx, file);
  for (const include of includes.values()) {
    const importPath = getImportPath(ctx, file, include);
    out += `import {\n  ${[...include.types.values()].join(
      ",\n  ",
    )}\n} from "${importPath}";\n`;
  }

  if (includes.size > 0) {
    out += "\n";
  }

  for (const fqtn of file.types) {
    out += writeType(ctx, file, fqtn);
  }

  if (ctx.usePrettier) {
    out = await prettier.format(out, ctx.prettierOptions);
  }

  const stream = fs.createWriteStream(file.path);
  stream.write(out);
  stream.end();
}

function getTSTypeName(
  ctx: TSContext,
  file: TSFile,
  fieldType: FieldType,
): string {
  switch (fieldType.c) {
    case "basic":
      switch (fieldType.type.sc) {
        case "bool":
          return "boolean";
        case "int":
        case "float":
          return "number";
        case "string":
          return "string";
      }
      break;
    case "vector":
      return `${getTSTypeName(ctx, file, fieldType.type)}[]`;
    case "custom": {
      const fqtn = fieldType.type.resolvedFqtn;
      return fqtn[fqtn.length - 1];
    }
  }
}

function writeType(ctx: TSContext, file: TSFile, fqtn: string): string {
  const type = ctx.schema.types.get(fqtn);
  if (!type) {
    throw new Error(`undefined type ${fqtn}`);
  }
  let out = `// ${type.relativeFbsFile}\n`;

  function writeEnum<T>(
    name: string,
    values: T[],
    valueFormatter: (value: T) => string | null,
  ) {
    const formattedValues = values.map(valueFormatter).filter(Boolean);
    if (formattedValues.length === 0) {
      return;
    }
    out += `export type ${name} =`;
    for (const value of formattedValues) {
      out += `\n  | ${value}`;
    }
    out += ";\n\n";
  }

  function unionValueFormatter(value: UnionValue) {
    const fqtn = value.typeRef.resolvedFqtn;
    const fqtnStr = fqtn.join(".");
    if (ctx.types.has(fqtnStr)) {
      return fqtn[fqtn.length - 1];
    } else {
      return null;
    }
  }

  function unionTagValueFormatter(value: UnionValue) {
    const name = unionValueFormatter(value);
    return name ? `"${name}"` : null;
  }

  function writeTable(type: TableType) {
    if (type.fields.length === 0) {
      out +=
        "// eslint-disable-next-line @typescript-eslint/no-empty-interface\n";
    }
    out += `export interface ${type.name} {`;
    for (const f of type.fields) {
      const typeName = getTSTypeName(ctx, file, f.type);
      const requiredField = isRequired(f.metadata);
      const typeSuffix = requiredField ? "" : "?";
      if (f.type.c === "custom") {
        const fqtn = f.type.type.resolvedFqtn.join(".");
        const resolvedType = ctx.schema.types.get(fqtn);
        if (!resolvedType) {
          throw new Error(`unknown type ${fqtn}`);
        }
        if (resolvedType.type === "union") {
          out += `\n  ${f.name}_type${typeSuffix}: ${getUnionTagTypeName(
            resolvedType.name,
          )};`;
        }
      }
      out += `\n  ${f.name}${typeSuffix}: ${typeName};`;

      const comments: string[] = [];
      if (f.defaultValue !== null) {
        comments.push(`default: ${f.defaultValue.value}`);
      }
      if (f.metadata !== null) {
        for (const attr of f.metadata) {
          if (attr.id === "optional" || attr.id === "required") {
            continue;
          }
          let comment = `${attr.id}`;
          if (attr.value) {
            comment += `: ${attr.value.value}`;
          }
          comments.push(comment);
        }
      }
      if (comments.length > 0) {
        out += ` // ${comments.join(", ")}`;
      }
    }
    out += "\n}\n\n";
  }

  switch (type.type) {
    case "enum":
      writeEnum(type.name, type.values, (value) => `"${value.id}"`);
      break;
    case "union":
      writeEnum(type.name, type.values, unionValueFormatter);
      writeEnum(
        getUnionTagTypeName(type.name),
        type.values,
        unionTagValueFormatter,
      );
      break;
    case "table":
      writeTable(type);
      break;
  }

  return out;
}
