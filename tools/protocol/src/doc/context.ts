import { Documentation } from "@/doc/types";
import { SchemaTypes } from "@/schema/types";

export interface DocSchemaFile {
  path: string;
  namespace: string;
  types: string[];
}

export interface DocContext {
  schema: SchemaTypes;
  doc: Documentation;
  schemasDir: string;
  pathsFile: string;
  tagsFile: string;
  schemaFiles: Map<string, DocSchemaFile>;
}
