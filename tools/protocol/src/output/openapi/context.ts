import { SchemaTypes } from "../../schema/types";
import { TypeFilter } from "../../filter/type-filter";
import { JSONSchema } from "../json-schema/types";
import { Document } from "yaml";

export type OpenAPIVersion = "3.1.0";

export interface OpenApiContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  baseUri: URL;
  openApiVersion: OpenAPIVersion;
  jsonSchema: Record<string, JSONSchema>;
  doc: Document;
  includeIds: boolean;
}
