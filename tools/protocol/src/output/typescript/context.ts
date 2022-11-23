import { SchemaTypes } from "../../schema/types";
import { TypeFilter } from "../../filter/type-filter";

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
}
