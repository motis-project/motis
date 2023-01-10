import { Options as PrettierOptions } from "prettier";

import { TypeFilter } from "@/filter/type-filter";
import { SchemaTypes } from "@/schema/types";

export interface TSFile {
  path: string;
  namespace: string;
  types: string[];
}

export interface TSContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  files: Map<string, TSFile>;
  types: Set<string>;
  header: string;
  outputDir: string;
  importBase: string | null;
  usePrettier: boolean;
  prettierOptions: PrettierOptions;
}
