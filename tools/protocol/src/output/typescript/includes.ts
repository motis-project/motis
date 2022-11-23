import { TSContext, TSFile } from "./context";
import { FieldType, TypeRef } from "../../schema/types";
import { getFilename } from "./filenames";

export interface TSInclude {
  filename: string;
  types: Set<string>;
}

export type TSIncludes = Map<string, TSInclude>;

export function collectIncludes(ctx: TSContext, file: TSFile): TSIncludes {
  const includes: TSIncludes = new Map();

  function handleTypeRef(ref: TypeRef) {
    const refFqtn = ref.resolvedFqtn.join(".");
    const refNamespace = ref.resolvedFqtn.slice(0, -1);
    const refNamespaceStr = refNamespace.join(".");
    if (refNamespaceStr === file.namespace || !ctx.types.has(refFqtn)) {
      return;
    }
    let include = includes.get(refNamespaceStr);
    if (!include) {
      include = { filename: getFilename(ctx, refNamespace), types: new Set() };
      includes.set(refNamespaceStr, include);
    }
    include.types.add(ref.resolvedFqtn[ref.resolvedFqtn.length - 1]);
  }

  function handleFieldType(ft: FieldType) {
    switch (ft.c) {
      case "basic":
        break;
      case "vector":
        handleFieldType(ft.type);
        break;
      case "custom":
        handleTypeRef(ft.type);
        break;
    }
  }

  for (const currentTypeName of file.types) {
    const currentType = ctx.schema.types.get(currentTypeName);
    if (!currentType) {
      throw new Error(`undefined type ${currentTypeName}`);
    }
    switch (currentType.type) {
      case "enum":
        break;
      case "union":
        for (const v of currentType.values) {
          handleTypeRef(v.typeRef);
        }
        break;
      case "table":
        for (const f of currentType.fields) {
          handleFieldType(f.type);
        }
        break;
    }
  }

  return includes;
}
