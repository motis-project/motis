import { SchemaTypes } from "../schema/types";
import { Documentation } from "./types";

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
  schemaFiles: Map<string, DocSchemaFile>;
}
