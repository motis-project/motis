export type JSONSchemaType =
  | "null"
  | "boolean"
  | "object"
  | "array"
  | "number"
  | "string"
  | "integer";

export type JSONLiteral = string | number | boolean | null;

export type JSONValue =
  | JSONLiteral
  | JSONValue[]
  | { [name: string]: JSONValue };

// not json schema, openapi extension
export interface JSONDiscriminator {
  propertyName: string;
  mapping: Record<string, string>;
}

// just a rough subset
export interface JSONSchema {
  $schema?: string;
  $id?: string;
  $comment?: string;
  $ref?: string;
  $defs?: Record<string, JSONSchema>;

  type?: JSONSchemaType | JSONSchemaType[];

  properties?: Record<string, JSONSchema>;
  required?: string[];
  additionalProperties?: boolean | JSONSchema;

  items?: JSONSchema;

  enum?: JSONLiteral[];

  allOf?: JSONSchema[];
  anyOf?: JSONSchema[];
  oneOf?: JSONSchema[];
  not?: JSONSchema;

  if?: JSONSchema;
  then?: JSONSchema;
  else?: JSONSchema;

  const?: JSONLiteral;

  format?: string;
  minimum?: number;
  maximum?: number;

  title?: string;
  description?: string;
  default?: JSONValue;
  examples?: JSONValue[];
  readOnly?: boolean;
  writeOnly?: boolean;
  deprecated?: boolean;

  // openapi extension
  discriminator?: JSONDiscriminator;
}
