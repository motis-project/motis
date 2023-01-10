import { Documentation } from "@/doc/types";
import { TypeFilter } from "@/filter/type-filter";
import { TSFile } from "@/output/typescript/context";
import { SchemaTypes } from "@/schema/types";

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
