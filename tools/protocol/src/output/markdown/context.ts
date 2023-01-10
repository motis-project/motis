import { Documentation } from "../../doc/types";
import { TypeFilter } from "../../filter/type-filter";
import { SchemaTypes } from "../../schema/types";
import { TSFile } from "../typescript/context";

export interface MarkdownFile {
  path: string;
  namespace: string;
  types: string[];
}

export interface MarkdownContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  doc: Documentation;
  files: Map<string, TSFile>;
  types: Set<string>;
  outputDir: string;
}
