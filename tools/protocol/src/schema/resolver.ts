import fs from "fs";
import * as path from "path";

import { AstFieldType, AstTopLevel } from "@/fbs/ast";
import { schema } from "@/fbs/parser/schema";
import {
  FieldType,
  SchemaType,
  SchemaTypes,
  TableType,
  TypeRef,
} from "@/schema/types";

interface ResolverContext {
  schema: SchemaTypes;
  files: Set<string>;
  rootDir: string;

  currentFile: string;
}

function getRelativeFbsFile(ctx: ResolverContext, fbsFile: string): string {
  return path.relative(ctx.rootDir, fbsFile).replace("\\", "/");
}

export function resolveSchemaTypes(
  rootDir: string,
  rootFile: string,
): SchemaTypes {
  const schema: SchemaTypes = {
    types: new Map<string, SchemaType>(),
    rootType: undefined,
  };
  const ctx: ResolverContext = {
    schema: schema,
    files: new Set<string>(),
    rootDir,
    currentFile: rootFile,
  };

  resolveFile(ctx, rootFile);
  console.log(`parsed ${ctx.files.size} files`);
  resolveTypes(ctx);
  console.log(`resolved ${ctx.schema.types.size} types`);

  return schema;
}

function resolveFile(ctx: ResolverContext, file: string) {
  if (ctx.files.has(file)) {
    return;
  }
  if (!fs.existsSync(file)) {
    throw new Error(`file not found: ${file}`);
  }
  ctx.currentFile = file;
  ctx.files.add(file);

  const fbs = fs.readFileSync(file, { encoding: "utf8" });
  const parseResult = schema.run(fbs);
  if (parseResult.isError) {
    console.log(`ERROR: failed to parse file: ${file}`);
    console.dir(parseResult, { depth: null });
    throw new Error(`failed to parse file: ${file}`);
  }

  const ast = parseResult.result;

  for (const elm of ast) {
    if (elm.t === "include") {
      const includeFile = path.resolve(ctx.rootDir, elm.value);
      resolveFile(ctx, includeFile);
    }
  }

  extractTypes(ctx, file, ast);
}

function namespaced(ns: string[], id: string): string[] {
  return [...ns, ...id.split(".")];
}

function verifyNewType(ctx: ResolverContext, fqtn: string[]) {
  const s = fqtn.join(".");
  if (ctx.schema.types.has(s)) {
    throw new Error(`type already defined: ${s}`);
  }
}

function extractType(currentNamespace: string[], id: string): TypeRef {
  return { input: id, currentNamespace, resolvedFqtn: [] };
}

function extractFieldType(
  ctx: ResolverContext,
  currentNamespace: string[],
  aft: AstFieldType,
): FieldType {
  switch (aft.c) {
    case "basic":
      return { c: "basic", type: aft.type };
    case "vector":
      return {
        c: "vector",
        type: extractFieldType(ctx, currentNamespace, aft.type),
      };
    case "custom":
      return {
        c: "custom",
        type: extractType(currentNamespace, aft.type),
      };
  }
}

function extractTypes(
  ctx: ResolverContext,
  fbsFile: string,
  ast: AstTopLevel[],
) {
  const relativeFbsFile = getRelativeFbsFile(ctx, fbsFile);
  let currentNamespace: string[] = [];
  ctx.currentFile = fbsFile;

  for (const elm of ast) {
    switch (elm.t) {
      case "namespace": {
        currentNamespace = elm.value.split(".");
        break;
      }
      case "enum": {
        const fqtn = namespaced(currentNamespace, elm.name);
        verifyNewType(ctx, fqtn);
        ctx.schema.types.set(fqtn.join("."), {
          name: elm.name,
          ns: currentNamespace,
          fbsFile,
          relativeFbsFile,
          metadata: elm.metadata,
          type: "enum",
          values: elm.values,
        });
        break;
      }
      case "union": {
        const fqtn = namespaced(currentNamespace, elm.name);
        verifyNewType(ctx, fqtn);
        ctx.schema.types.set(fqtn.join("."), {
          name: elm.name,
          ns: currentNamespace,
          fbsFile,
          relativeFbsFile,
          metadata: elm.metadata,
          type: "union",
          values: elm.values.map((a) => {
            return {
              typeRef: extractType(currentNamespace, a.id),
              value: a.value,
              metadata: a.metadata,
            };
          }),
        });
        break;
      }
      case "table": {
        const fqtn = namespaced(currentNamespace, elm.name);
        verifyNewType(ctx, fqtn);
        ctx.schema.types.set(fqtn.join("."), {
          name: elm.name,
          ns: currentNamespace,
          fbsFile,
          relativeFbsFile,
          metadata: elm.metadata,
          type: "table",
          isStruct: elm.isStruct,
          fields: elm.fields.map((a) => {
            return {
              name: a.name,
              type: extractFieldType(ctx, currentNamespace, a.type),
              defaultValue: a.defaultValue,
              metadata: a.metadata,
            };
          }),
          usedInUnion: false,
          usedInTable: false,
        });
        break;
      }
      case "attribute": {
        break;
      }
      case "root_type": {
        ctx.schema.rootType = extractType(currentNamespace, elm.type.type);
        break;
      }
      default:
        break;
    }
  }
}

function resolveType(ctx: ResolverContext, ref: TypeRef) {
  const prefix = [...ref.currentNamespace];

  while (ref.resolvedFqtn.length === 0) {
    const fqtn = [...prefix, ...ref.input.split(".")];
    if (ctx.schema.types.has(fqtn.join("."))) {
      ref.resolvedFqtn = fqtn;
    } else if (prefix.length > 0) {
      prefix.pop();
    } else {
      throw new Error(
        `unknown type: ${ref.input}, referenced in ${ctx.currentFile}`,
      );
    }
  }
}

function markResolvedType(
  ctx: ResolverContext,
  ref: TypeRef,
  fn: (resolved: TableType) => void,
) {
  const resolved = ctx.schema.types.get(ref.resolvedFqtn.join("."));
  if (resolved && resolved.type === "table") {
    fn(resolved);
  }
}

function resolveFieldType(ctx: ResolverContext, ft: FieldType) {
  switch (ft.c) {
    case "basic":
      break;
    case "vector":
      resolveFieldType(ctx, ft.type);
      break;
    case "custom": {
      resolveType(ctx, ft.type);
      markResolvedType(
        ctx,
        ft.type,
        (resolved) => (resolved.usedInTable = true),
      );
      break;
    }
  }
}

function resolveTypes(ctx: ResolverContext) {
  for (const t of ctx.schema.types.values()) {
    switch (t.type) {
      case "enum":
        break;
      case "union":
        for (const uv of t.values) {
          resolveType(ctx, uv.typeRef);
          markResolvedType(
            ctx,
            uv.typeRef,
            (resolved) => (resolved.usedInUnion = true),
          );
        }
        break;
      case "table":
        for (const f of t.fields) {
          resolveFieldType(ctx, f.type);
        }
        break;
    }
  }
  if (ctx.schema.rootType) {
    resolveType(ctx, ctx.schema.rootType);
    markResolvedType(
      ctx,
      ctx.schema.rootType,
      (resolved) => (resolved.usedInTable = true),
    );
  }
}
