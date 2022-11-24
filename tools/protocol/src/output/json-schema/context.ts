import { SchemaTypes } from "../../schema/types";
import { TypeFilter } from "../../filter/type-filter";

export interface JSContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  baseUri: URL;
  jsonSchema: Map<string, any>;
  strictIntTypes: boolean;
  strictUnions: boolean;
}
