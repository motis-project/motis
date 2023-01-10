import { TypeFilter } from "../../filter/type-filter";
import { SchemaTypes } from "../../schema/types";
import { JSONSchema } from "./types";

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
