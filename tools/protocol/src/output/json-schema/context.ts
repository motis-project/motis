import { TypeFilter } from "@/filter/type-filter";
import { JSONSchema } from "@/output/json-schema/types";
import { SchemaTypes } from "@/schema/types";

export interface JSContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  baseUri: URL;
  jsonSchema: Map<string, JSONSchema>;
  strictIntTypes: boolean;
  numberFormats: boolean;
  strictUnions: boolean;
  getRefUrl: (fqtn: string[]) => string;
}
