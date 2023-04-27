import { TypeFilter } from "@/filter/type-filter";
import { JSONSchema } from "@/output/json-schema/types";
import { SchemaTypes } from "@/schema/types";

export interface JSContext {
  schema: SchemaTypes;
  typeFilter: TypeFilter;
  baseUri: URL;
  jsonSchema: Map<string, JSONSchema>;
  taggedToUntaggedType: Map<string, string>; // tagged -> untagged fqtn
  untaggedToTaggedType: Map<string, string>; // tagged -> untagged fqtn
  strictIntTypes: boolean;
  numberFormats: boolean;
  strictUnions: boolean;
  typesInUnions: boolean;
  getRefUrl: (fqtn: string[]) => string;
  getTaggedType: (fqtn: string[]) => string[];
  typeKey: string;
  includeOpenApiDiscriminators: boolean;
  constAsEnum: boolean;
  explicitAdditionalProperties: boolean;
}
